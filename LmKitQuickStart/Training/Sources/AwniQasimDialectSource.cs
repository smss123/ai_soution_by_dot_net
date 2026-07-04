using System.Text;
using System.Text.Json;
using HtmlAgilityPack;
using LMKit.Model;
using LMKit.TextGeneration.Chat;

namespace LmKitQuickStart.Training.Sources;

/// <summary>
/// Loads all training data from TrainingData/awni_qasim_*.json files.
/// Edit those files to add/update content — no code changes needed.
///
/// Files:
///   awni_qasim_vocabulary.json    — مفردات
///   awni_qasim_proverbs.json      — أمثال
///   awni_qasim_expressions.json   — تعابير ثقافية
///   awni_qasim_grammar.json       — قواعد اللهجة
///   awni_qasim_domain_terms.json  — مصطلحات متخصصة
/// </summary>
public sealed class AwniQasimDialectSource : ITrainingSource
{
    public string Name => "Awni Al-Sharif Qasim — قاموس اللهجة العامية السودانية";

    private static readonly string DataFolder = Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "TrainingData"));

    private const string SystemPrompt =
        "أنت Xprema، مساعد ذكاء اصطناعي يفهم اللهجة السودانية العامية بعمق ويستند " +
        "إلى مرجع دكتور عوني الشريف قاسم «قاموس اللهجة العامية السودانية».";

    // Wikipedia sources (still fetched from web, no hardcoding)
    private static readonly (string Topic, string Url)[] WebSources =
    [
        ("عوني الشريف قاسم",
         "https://ar.wikipedia.org/wiki/%D8%B9%D9%88%D9%86%D9%8A_%D8%A7%D9%84%D8%B4%D8%B1%D9%8A%D9%81_%D9%82%D8%A7%D8%B3%D9%85"),
        ("اللهجة السودانية",
         "https://ar.wikipedia.org/wiki/%D8%A7%D9%84%D9%84%D9%87%D8%AC%D8%A9_%D8%A7%D9%84%D8%B3%D9%88%D8%AF%D8%A7%D9%86%D9%8A%D8%A9"),
        ("أمثال سودانية",
         "https://ar.wikipedia.org/wiki/%D8%A3%D9%85%D8%AB%D8%A7%D9%84_%D8%B3%D9%88%D8%AF%D8%A7%D9%86%D9%8A%D8%A9"),
        ("الثقافة السودانية",
         "https://ar.wikipedia.org/wiki/%D8%AB%D9%82%D8%A7%D9%81%D8%A9_%D8%A7%D9%84%D8%B3%D9%88%D8%AF%D8%A7%D9%86"),
        ("الموسيقى السودانية",
         "https://ar.wikipedia.org/wiki/%D9%85%D9%88%D8%B3%D9%8A%D9%82%D9%89_%D8%B3%D9%88%D8%AF%D8%A7%D9%86%D9%8A%D8%A9"),
    ];

    public async Task<IReadOnlyList<ChatHistory>> GetSamplesAsync(LM baseModel, CancellationToken ct = default)
    {
        var samples = new List<ChatHistory>();

        await AddWebSamplesAsync(samples, baseModel, ct);
        AddVocabSamples(samples, baseModel);
        AddProverbSamples(samples, baseModel);
        AddExpressionSamples(samples, baseModel);
        AddGrammarSamples(samples, baseModel);
        AddDomainTermSamples(samples, baseModel);

        Console.WriteLine($"  [{Name}] {samples.Count} total samples.");
        return samples;
    }

    // ── Loaders ────────────────────────────────────────────────────

    private static T LoadJson<T>(string fileName) where T : class, new()
    {
        string path = Path.Combine(DataFolder, fileName);
        if (!File.Exists(path))
        {
            Console.WriteLine($"  [Warning] {fileName} not found — skipping.");
            return new T();
        }
        string json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<T>(json,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? new T();
    }

    // ── Section builders ───────────────────────────────────────────

    private void AddVocabSamples(List<ChatHistory> samples, LM baseModel)
    {
        var entries = LoadJson<List<VocabEntry>>("awni_qasim_vocabulary.json");
        if (entries.Count == 0) return;

        foreach (var chunk in entries.Chunk(10))
        {
            var sb = new StringBuilder("مفردات من قاموس دكتور عوني الشريف قاسم:\n\n");
            foreach (var e in chunk)
                sb.AppendLine($"• {e.Word} ← {e.Meaning}\n  مثال: {e.Example}");
            Add(samples, baseModel, "أعطني مفردات سودانية مع معانيها وأمثلتها.", sb.ToString().Trim());
        }

        foreach (var e in entries)
            Add(samples, baseModel,
                $"شنو معنى «{e.Word}» بالسوداني؟",
                $"«{e.Word}» تعني: {e.Meaning}.\nمثال: {e.Example}");
    }

    private void AddProverbSamples(List<ChatHistory> samples, LM baseModel)
    {
        var entries = LoadJson<List<ProverbEntry>>("awni_qasim_proverbs.json");
        if (entries.Count == 0) return;

        var allSb = new StringBuilder("أمثال شعبية سودانية مع معانيها:\n\n");
        foreach (var e in entries)
            allSb.AppendLine($"◆ «{e.Proverb}»\n   المعنى: {e.Meaning}\n   الاستخدام: {e.Context}\n");
        Add(samples, baseModel, "أعطني أمثال شعبية سودانية مع الشرح", allSb.ToString().Trim());

        foreach (var e in entries)
            Add(samples, baseModel,
                $"فسّر لي المثل السوداني: «{e.Proverb}»",
                $"المثل «{e.Proverb}» يعني:\n{e.Meaning}\n\nيُستخدم في: {e.Context}");
    }

    private void AddExpressionSamples(List<ChatHistory> samples, LM baseModel)
    {
        var entries = LoadJson<List<ExpressionEntry>>("awni_qasim_expressions.json");
        if (entries.Count == 0) return;

        foreach (var chunk in entries.Chunk(8))
        {
            var sb = new StringBuilder("تعابير وعبارات سودانية:\n\n");
            foreach (var e in chunk)
                sb.AppendLine($"• «{e.Expression}» → {e.Meaning}\n  استخدام: {e.Usage}");
            Add(samples, baseModel, "أعطني تعابير سودانية شائعة مع معانيها", sb.ToString().Trim());
        }
    }

    private void AddGrammarSamples(List<ChatHistory> samples, LM baseModel)
    {
        var entries = LoadJson<List<GrammarEntry>>("awni_qasim_grammar.json");
        if (entries.Count == 0) return;

        foreach (var e in entries)
        {
            var sb = new StringBuilder($"{e.Explanation}\n\nأمثلة:\n");
            foreach (var ex in e.Examples) sb.AppendLine($"  • {ex}");
            Add(samples, baseModel, $"اشرح لي قاعدة «{e.Pattern}» في اللهجة السودانية", sb.ToString().Trim());
        }
    }

    private void AddDomainTermSamples(List<ChatHistory> samples, LM baseModel)
    {
        var domains = LoadJson<List<DomainEntry>>("awni_qasim_domain_terms.json");
        if (domains.Count == 0) return;

        foreach (var domain in domains)
        {
            var sb = new StringBuilder($"مصطلحات سودانية في مجال {domain.Domain}:\n\n");
            foreach (var t in domain.Terms)
                sb.AppendLine($"• {t.Term}: {t.Meaning}");
            Add(samples, baseModel, $"أعطني مصطلحات سودانية خاصة بـ {domain.Domain}", sb.ToString().Trim());
        }
    }

    private static async Task AddWebSamplesAsync(
        List<ChatHistory> samples, LM baseModel, CancellationToken ct)
    {
        using var http = CreateHttpClient();
        foreach (var (topic, url) in WebSources)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                Console.Write($"  Fetching {topic}... ");
                string html = await http.GetStringAsync(url, ct);
                string text = ExtractWikipediaText(html);
                if (string.IsNullOrWhiteSpace(text) || text.Length < 200)
                { Console.WriteLine("(قصير جداً)"); continue; }

                Add(samples, baseModel, $"أعطني معلومات عن: {topic}", text.Trim());
                Console.WriteLine("OK");
                await Task.Delay(500, ct);
            }
            catch (OperationCanceledException ex) when (ex is not TaskCanceledException || ct.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex) { Console.WriteLine($"ERROR: {ex.Message}"); }
        }
    }

    // ── Helpers ────────────────────────────────────────────────────

    private static void Add(List<ChatHistory> samples, LM baseModel, string user, string assistant)
    {
        var h = new ChatHistory(baseModel);
        h.AddMessage(AuthorRole.System,    SystemPrompt);
        h.AddMessage(AuthorRole.User,      user);
        h.AddMessage(AuthorRole.Assistant, assistant);
        samples.Add(h);
    }

    private static string ExtractWikipediaText(string html)
    {
        var doc = new HtmlDocument();
        doc.LoadHtml(html);
        foreach (string tag in (string[])["script", "style", "nav", "footer", "header", "table"])
        {
            var nodes = doc.DocumentNode.SelectNodes($"//{tag}");
            if (nodes is not null) foreach (var n in nodes) n.Remove();
        }
        var content = doc.DocumentNode.SelectSingleNode("//div[@id='mw-content-text']")
                   ?? doc.DocumentNode.SelectSingleNode("//main")
                   ?? doc.DocumentNode.SelectSingleNode("//body");
        if (content is null) return string.Empty;
        var sb = new StringBuilder();
        foreach (var node in content.DescendantsAndSelf())
        {
            if (node.NodeType != HtmlNodeType.Text) continue;
            string text = HtmlEntity.DeEntitize(node.InnerText).Trim();
            if (!string.IsNullOrWhiteSpace(text) && text.Length > 3) sb.AppendLine(text);
        }
        string result = sb.ToString();
        return result.Length > 4000 ? result[..4000] : result;
    }

    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("Mozilla/5.0 (compatible; XpremaTrainer/1.0)");
        client.DefaultRequestHeaders.AcceptLanguage.ParseAdd("ar,en;q=0.5");
        return client;
    }

    // ── JSON models ────────────────────────────────────────────────

    private sealed record VocabEntry(string Word, string Meaning, string Example);
    private sealed record ProverbEntry(string Proverb, string Meaning, string Context);
    private sealed record ExpressionEntry(string Expression, string Meaning, string Usage);
    private sealed record GrammarEntry(string Pattern, string Explanation, List<string> Examples);
    private sealed record DomainEntry(string Domain, List<TermEntry> Terms);
    private sealed record TermEntry(string Term, string Meaning);
}
