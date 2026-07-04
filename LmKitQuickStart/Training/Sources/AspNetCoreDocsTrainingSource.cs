using LMKit.Model;
using LMKit.TextGeneration.Chat;

namespace LmKitQuickStart.Training.Sources;

/// <summary>
/// Generates training samples from official Microsoft ASP.NET Core documentation.
/// </summary>
public sealed class  AspNetCoreDocsTrainingSource : ITrainingSource
{
    public string Name => "ASP.NET Core Docs";

    private const string SystemPrompt =
        "You are Xprema, an expert AI assistant for ASP.NET Core web development.";

    private static readonly (string Topic, string Url)[] Pages =
    [
        ("ASP.NET Core overview",                  "https://learn.microsoft.com/en-us/aspnet/core/introduction-to-aspnet-core"),
        ("ASP.NET Core middleware",                "https://learn.microsoft.com/en-us/aspnet/core/fundamentals/middleware/"),
        ("ASP.NET Core dependency injection",      "https://learn.microsoft.com/en-us/aspnet/core/fundamentals/dependency-injection"),
        ("ASP.NET Core configuration",             "https://learn.microsoft.com/en-us/aspnet/core/fundamentals/configuration/"),
        ("ASP.NET Core logging",                   "https://learn.microsoft.com/en-us/aspnet/core/fundamentals/logging/"),
        ("ASP.NET Core routing",                   "https://learn.microsoft.com/en-us/aspnet/core/fundamentals/routing"),
        ("ASP.NET Core authentication",            "https://learn.microsoft.com/en-us/aspnet/core/security/authentication/"),
        ("ASP.NET Core authorization",             "https://learn.microsoft.com/en-us/aspnet/core/security/authorization/introduction"),
        ("ASP.NET Core minimal APIs",              "https://learn.microsoft.com/en-us/aspnet/core/fundamentals/minimal-apis"),
        ("ASP.NET Core Web API controllers",       "https://learn.microsoft.com/en-us/aspnet/core/web-api/"),
        ("ASP.NET Core Entity Framework Core",     "https://learn.microsoft.com/en-us/aspnet/core/data/ef-mvc/intro"),
        ("ASP.NET Core Razor Pages",               "https://learn.microsoft.com/en-us/aspnet/core/razor-pages/"),
        ("ASP.NET Core Blazor overview",           "https://learn.microsoft.com/en-us/aspnet/core/blazor/"),
        ("ASP.NET Core SignalR",                   "https://learn.microsoft.com/en-us/aspnet/core/signalr/introduction"),
        ("ASP.NET Core gRPC",                      "https://learn.microsoft.com/en-us/aspnet/core/grpc/"),
        ("ASP.NET Core caching",                   "https://learn.microsoft.com/en-us/aspnet/core/performance/caching/overview"),
        ("ASP.NET Core health checks",             "https://learn.microsoft.com/en-us/aspnet/core/host-and-deploy/health-checks"),
        ("ASP.NET Core background services",       "https://learn.microsoft.com/en-us/aspnet/core/fundamentals/host/hosted-services"),
        ("ASP.NET Core testing",                   "https://learn.microsoft.com/en-us/aspnet/core/test/integration-tests"),
        ("ASP.NET Core deployment",                "https://learn.microsoft.com/en-us/aspnet/core/host-and-deploy/"),
    ];

    public async Task<IReadOnlyList<ChatHistory>> GetSamplesAsync(LM baseModel, CancellationToken ct = default)
    {
        var samples = new List<ChatHistory>();
        using var http = CreateHttpClient();

        foreach (var (topic, url) in Pages)
        {
            ct.ThrowIfCancellationRequested();

            try
            {
                string html = await http.GetStringAsync(url, ct);
                string text = HtmlTextExtractor.Extract(html);
                if (string.IsNullOrWhiteSpace(text)) continue;

                var history = new ChatHistory(baseModel);
                history.AddMessage(AuthorRole.System,    SystemPrompt);
                history.AddMessage(AuthorRole.User,      $"Explain {topic} in detail.");
                history.AddMessage(AuthorRole.Assistant, text.Trim());

                samples.Add(history);
                await Task.Delay(300, ct);
            }
            catch (OperationCanceledException ex) when (ex is not TaskCanceledException || ct.IsCancellationRequested) { throw; }
            catch (Exception ex) { Console.WriteLine($"  [ASP.NET] Skip {topic}: {ex.Message}"); }
        }

        return samples;
    }

    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("Mozilla/5.0 (compatible; XpremaTrainer/1.0)");
        return client;
    }
}

