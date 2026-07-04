using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using HtmlAgilityPack;
using LMKit.Model;
using LMKit.TextGeneration.Chat;

namespace LmKitQuickStart.Training.Sources;

/// <summary>
/// Crawls Sudanese dialect content from the internet.
/// Starts from seed URLs in TrainingData/sudanese_web_seeds.json,
/// discovers additional links on each domain, up to MaxUrls total.
/// Discovered URLs are saved to TrainingData/sudanese_discovered_urls.json
/// so re-runs skip already-crawled pages.
/// </summary>
public sealed class SudaneseWebCrawlerSource : ITrainingSource
{
    public string Name => "Sudanese Web Crawler";

    private const int MaxUrls         = 1000;
    private const int CrawlDelayMs    = 500;
    private const int MinTextLength   = 150;
    private const int MaxTextLength   = 5000;
    private const int MaxLinksPerPage = 30;

    private static readonly string DataFolder = Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "TrainingData"));

    private static readonly string SeedsFile     = Path.Combine(DataFolder, "sudanese_web_seeds.json");
    private static readonly string DiscoveredFile = Path.Combine(DataFolder, "sudanese_discovered_urls.json");

    // Domains we are allowed to follow links within
    private static readonly HashSet<string> AllowedDomains =
    [
        "iqraweb.net",
        "medameek.com",
        "sudanile.com",
        "andariya.com",
        "storiesrealistic.com",
        "sudaneseshortstorieswriters.blogspot.com",
        "ar.wikipedia.org",
        "alrakoba.net",
        "sudaress.com",
    ];

    // Arabic keywords that confirm relevance of a page
    private static readonly string[] RelevanceKeywords =
    [
        "سوداني", "السودان", "لهجة", "عامية", "مفردات", "أمثال",
        "حوار", "قصة", "تراث", "ثقافة", "خرطوم", "كلام",
    ];

    private const string SystemPrompt =
        "أنت Xprema، مساعد ذكاء اصطناعي يفهم اللهجة السودانية ويعرف ثقافة وتراث السودان.";

    // ── Public API ─────────────────────────────────────────────────

    public async Task<IReadOnlyList<ChatHistory>> GetSamplesAsync(LM baseModel, CancellationToken ct = default)
    {
        var seeds      = LoadSeeds();
        var discovered = LoadDiscovered();
        var crawled    = new HashSet<string>(discovered.Select(d => d.Url), StringComparer.OrdinalIgnoreCase);
        var queue      = new Queue<SeedEntry>(seeds.Where(s => !crawled.Contains(s.Url)));

        Console.WriteLine($"  Seeds: {seeds.Count}  Already crawled: {crawled.Count}  Queue: {queue.Count}");

        using var http = CreateHttpClient();
        var samples    = new List<ChatHistory>();
        int newPages   = 0;

        while (queue.Count > 0 && crawled.Count < MaxUrls)
        {
            ct.ThrowIfCancellationRequested();

            var entry = queue.Dequeue();
            if (crawled.Contains(entry.Url)) continue;
            crawled.Add(entry.Url);

            try
            {
                Console.Write($"  [{crawled.Count}/{MaxUrls}] {entry.Label.Truncate(40)}... ");
                string html = await http.GetStringAsync(entry.Url, ct);
                string text = ExtractText(html);

                if (text.Length < MinTextLength)
                { Console.WriteLine("(قصير)"); continue; }

                // Trim to max length
                if (text.Length > MaxTextLength) text = text[..MaxTextLength];

                // Create training sample
                var history = new ChatHistory(baseModel);
                history.AddMessage(AuthorRole.System,    SystemPrompt);
                history.AddMessage(AuthorRole.User,      BuildQuestion(entry.Label, text));
                history.AddMessage(AuthorRole.Assistant, text.Trim());
                samples.Add(history);

                // Save to discovered list
                discovered.Add(entry);
                newPages++;
                Console.WriteLine($"OK ({text.Length} حرف)");

                // Discover new links from this page
                if (crawled.Count < MaxUrls)
                {
                    var newLinks = DiscoverLinks(html, entry.Url, entry.Domain, crawled);
                    foreach (var link in newLinks.Take(MaxLinksPerPage))
                        queue.Enqueue(link);
                }

                // Save progress every 50 pages
                if (newPages % 50 == 0)
                    SaveDiscovered(discovered);

                await Task.Delay(CrawlDelayMs, ct);
            }
            catch (OperationCanceledException ex) when (ex is not TaskCanceledException || ct.IsCancellationRequested)
            {
                throw; // only rethrow if user cancelled, not HTTP timeout
            }
            catch (Exception ex) { Console.WriteLine($"ERROR: {ex.Message}"); }
        }

        SaveDiscovered(discovered);
        Console.WriteLine($"\n  Crawled {newPages} new pages. Total discovered: {discovered.Count}. Samples: {samples.Count}");
        return samples;
    }

    // ── Link discovery ─────────────────────────────────────────────

    private static List<SeedEntry> DiscoverLinks(string html, string baseUrl, string domain, HashSet<string> visited)
    {
        var doc = new HtmlDocument();
        doc.LoadHtml(html);

        var links   = new List<SeedEntry>();
        var baseUri = new Uri(baseUrl);

        var anchors = doc.DocumentNode.SelectNodes("//a[@href]");
        if (anchors is null) return links;

        foreach (var anchor in anchors)
        {
            string href = anchor.GetAttributeValue("href", "").Trim();
            if (string.IsNullOrWhiteSpace(href) || href.StartsWith('#') || href.StartsWith("mailto:"))
                continue;

            string absoluteUrl = ResolveUrl(baseUri, href);
            if (absoluteUrl is null) continue;
            if (visited.Contains(absoluteUrl)) continue;

            // Only follow links within allowed domains
            if (!Uri.TryCreate(absoluteUrl, UriKind.Absolute, out var uri)) continue;
            if (!AllowedDomains.Contains(uri.Host.Replace("www.", ""))) continue;

            // Skip non-content URLs
            if (IsResourceUrl(absoluteUrl)) continue;

            string label = anchor.InnerText.Trim().Truncate(60);
            if (string.IsNullOrWhiteSpace(label)) label = uri.PathAndQuery;

            // Prioritize pages with Arabic dialect keywords in URL or anchor text
            if (IsRelevant(absoluteUrl + " " + label))
                links.Insert(0, new SeedEntry(uri.Host.Replace("www.", ""), absoluteUrl, label));
            else
                links.Add(new SeedEntry(uri.Host.Replace("www.", ""), absoluteUrl, label));
        }

        return links;
    }

    private static string? ResolveUrl(Uri baseUri, string href)
    {
        try
        {
            var resolved = new Uri(baseUri, href);
            return resolved.Scheme is "http" or "https" ? resolved.ToString() : null;
        }
        catch { return null; }
    }

    private static bool IsResourceUrl(string url) =>
        url.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
        url.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
        url.EndsWith(".gif", StringComparison.OrdinalIgnoreCase) ||
        url.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase) ||
        url.EndsWith(".mp3", StringComparison.OrdinalIgnoreCase) ||
        url.EndsWith(".mp4", StringComparison.OrdinalIgnoreCase) ||
        url.Contains("/tag/") ||
        url.Contains("/feed/") ||
        url.Contains("?replytocom") ||
        url.Contains("/wp-login");

    private static bool IsRelevant(string text) =>
        RelevanceKeywords.Any(k => text.Contains(k, StringComparison.OrdinalIgnoreCase));

    // ── Text extraction ────────────────────────────────────────────

    private static string ExtractText(string html)
    {
        var doc = new HtmlDocument();
        doc.LoadHtml(html);

        foreach (string tag in (string[])["script", "style", "nav", "footer", "header", "aside", "form"])
        {
            var nodes = doc.DocumentNode.SelectNodes($"//{tag}");
            if (nodes is not null) foreach (var n in nodes) n.Remove();
        }

        // Try to find main content
        var content =
            doc.DocumentNode.SelectSingleNode("//article") ??
            doc.DocumentNode.SelectSingleNode("//div[@class='entry-content']") ??
            doc.DocumentNode.SelectSingleNode("//div[@class='post-content']") ??
            doc.DocumentNode.SelectSingleNode("//div[@class='content']") ??
            doc.DocumentNode.SelectSingleNode("//div[@id='content']") ??
            doc.DocumentNode.SelectSingleNode("//div[@id='mw-content-text']") ??
            doc.DocumentNode.SelectSingleNode("//main") ??
            doc.DocumentNode.SelectSingleNode("//body");

        if (content is null) return string.Empty;

        var sb = new StringBuilder();
        foreach (var node in content.DescendantsAndSelf())
        {
            if (node.NodeType != HtmlNodeType.Text) continue;
            string text = HtmlEntity.DeEntitize(node.InnerText).Trim();
            if (!string.IsNullOrWhiteSpace(text) && text.Length > 5)
                sb.AppendLine(text);
        }

        // Collapse multiple blank lines
        string result = Regex.Replace(sb.ToString(), @"\n{3,}", "\n\n");
        return result.Trim();
    }

    // ── Question builder ───────────────────────────────────────────

    private static string BuildQuestion(string label, string text)
    {
        // Generate contextually appropriate questions based on the content type
        if (label.Contains("مثل") || label.Contains("أمثال"))
            return "أعطني أمثال شعبية سودانية مع شرح معانيها.";
        if (label.Contains("قاموس") || label.Contains("مفردات") || label.Contains("كلمات"))
            return "أعطني مفردات من اللهجة السودانية العامية مع شرح معانيها.";
        if (label.Contains("قصة") || label.Contains("حكاية") || label.Contains("رواية"))
            return "احكلي قصة سودانية باللهجة السودانية.";
        if (label.Contains("لهجة") || label.Contains("عامية"))
            return "اشرح لي اللهجة السودانية العامية.";
        if (label.Contains("ثقافة") || label.Contains("تراث"))
            return "حدثني عن الثقافة والتراث السوداني.";
        if (label.Contains("تاريخ"))
            return "حدثني عن تاريخ السودان.";
        if (label.Contains("موسيقى") || label.Contains("فنون"))
            return "حدثني عن الفنون والموسيقى السودانية.";
        if (label.Contains("أكل") || label.Contains("طعام") || label.Contains("مطبخ"))
            return "حدثني عن الأكل والطبخ السوداني.";

        return $"أعطني معلومات عن: {label}";
    }

    // ── File I/O ───────────────────────────────────────────────────

    private static List<SeedEntry> LoadSeeds()
    {
        if (!File.Exists(SeedsFile)) return [];
        string json = File.ReadAllText(SeedsFile);
        return JsonSerializer.Deserialize<List<SeedEntry>>(json,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? [];
    }

    private static List<SeedEntry> LoadDiscovered()
    {
        if (!File.Exists(DiscoveredFile)) return [];
        try
        {
            string json = File.ReadAllText(DiscoveredFile);
            return JsonSerializer.Deserialize<List<SeedEntry>>(json,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? [];
        }
        catch { return []; }
    }

    private static void SaveDiscovered(List<SeedEntry> discovered)
    {
        string json = JsonSerializer.Serialize(discovered,
            new JsonSerializerOptions { WriteIndented = false, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
        File.WriteAllText(DiscoveredFile, json, Encoding.UTF8);
    }

    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
        client.DefaultRequestHeaders.UserAgent.ParseAdd(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36");
        client.DefaultRequestHeaders.AcceptLanguage.ParseAdd("ar,en;q=0.5");
        client.DefaultRequestHeaders.Accept.ParseAdd("text/html,application/xhtml+xml");
        return client;
    }

    // ── Models ─────────────────────────────────────────────────────

    private sealed record SeedEntry(string Domain, string Url, string Label);
}

// ── Extension ─────────────────────────────────────────────────────

internal static class StringExtensions
{
    public static string Truncate(this string s, int maxLength) =>
        s.Length <= maxLength ? s : s[..maxLength] + "…";
}
