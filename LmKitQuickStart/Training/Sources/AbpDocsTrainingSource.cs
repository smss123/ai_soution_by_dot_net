using LMKit.Model;
using LMKit.TextGeneration.Chat;
using LmKitQuickStart.Rags;

namespace LmKitQuickStart.Training.Sources;

/// <summary>
/// Generates training samples from ABP Framework documentation.
/// Reuses the AbpDocsKnowledgeBase crawling infrastructure.
/// Each doc page becomes a single Q&A training sample.
/// </summary>
public sealed class AbpDocsTrainingSource : ITrainingSource
{
    public string Name => "ABP Framework Docs";

    private const string SystemPrompt =
        "You are Xprema, an expert AI assistant for ABP Framework, C#, and ASP.NET Core development.";

    public async Task<IReadOnlyList<ChatHistory>> GetSamplesAsync(LM baseModel, CancellationToken ct = default)
    {
        var samples = new List<ChatHistory>();
        using var http = CreateHttpClient();

        foreach (var (section, url) in AbpDocsKnowledgeBase.DocPages)
        {
            ct.ThrowIfCancellationRequested();

            try
            {
                string text = await FetchTextAsync(http, url, ct);
                if (string.IsNullOrWhiteSpace(text)) continue;

                string topic = SectionToTopic(section);

                var history = new ChatHistory(baseModel);
                history.AddMessage(AuthorRole.System,    SystemPrompt);
                history.AddMessage(AuthorRole.User,      $"Explain {topic} in ABP Framework.");
                history.AddMessage(AuthorRole.Assistant, text.Trim());

                samples.Add(history);
                await Task.Delay(300, ct);
            }
            catch (OperationCanceledException ex) when (ex is not TaskCanceledException || ct.IsCancellationRequested) { throw; }
            catch (Exception ex) { Console.WriteLine($"  [ABP] Skip {section}: {ex.Message}"); }
        }

        return samples;
    }

    private static async Task<string> FetchTextAsync(HttpClient http, string url, CancellationToken ct)
    {
        string html = await http.GetStringAsync(url, ct);
        return HtmlTextExtractor.Extract(html);
    }

    private static string SectionToTopic(string section) =>
        section.Replace('/', ' ').Replace('-', ' ');

    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("Mozilla/5.0 (compatible; XpremaTrainer/1.0)");
        return client;
    }
}

