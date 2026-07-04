using LMKit.Model;
using LMKit.TextGeneration.Chat;

namespace LmKitQuickStart.Training.Sources;

/// <summary>
/// Generates training samples from official Microsoft C# documentation.
/// </summary>
public sealed class CSharpDocsTrainingSource : ITrainingSource
{
    public string Name => "C# Docs";

    private const string SystemPrompt =
        "You are Xprema, an expert AI assistant for C# and .NET development.";

    private static readonly (string Topic, string Url)[] Pages =
    [
        ("C# language overview",              "https://learn.microsoft.com/en-us/dotnet/csharp/tour-of-csharp/"),
        ("C# types and variables",            "https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/types/"),
        ("C# classes and objects",            "https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/types/classes"),
        ("C# interfaces",                     "https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/types/interfaces"),
        ("C# generics",                       "https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/types/generics"),
        ("C# records",                        "https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/types/records"),
        ("C# LINQ",                           "https://learn.microsoft.com/en-us/dotnet/csharp/linq/"),
        ("C# async and await",                "https://learn.microsoft.com/en-us/dotnet/csharp/asynchronous-programming/"),
        ("C# delegates and events",           "https://learn.microsoft.com/en-us/dotnet/csharp/programming-guide/delegates/"),
        ("C# exception handling",             "https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/exceptions/"),
        ("C# pattern matching",               "https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/functional/pattern-matching"),
        ("C# nullable reference types",       "https://learn.microsoft.com/en-us/dotnet/csharp/nullable-references"),
        ("C# dependency injection .NET",      "https://learn.microsoft.com/en-us/dotnet/core/extensions/dependency-injection"),
        ("C# collections and data structures","https://learn.microsoft.com/en-us/dotnet/standard/collections/"),
        ("C# string handling",                "https://learn.microsoft.com/en-us/dotnet/csharp/programming-guide/strings/"),
        ("C# file IO",                        "https://learn.microsoft.com/en-us/dotnet/standard/io/"),
        ("C# memory and spans",               "https://learn.microsoft.com/en-us/dotnet/standard/memory-and-spans/"),
        ("C# unit testing",                   "https://learn.microsoft.com/en-us/dotnet/core/testing/"),
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
                history.AddMessage(AuthorRole.User,      $"Explain {topic} with examples.");
                history.AddMessage(AuthorRole.Assistant, text.Trim());

                samples.Add(history);
                await Task.Delay(300, ct);
            }
            catch (OperationCanceledException ex) when (ex is not TaskCanceledException || ct.IsCancellationRequested) { throw; }
            catch (Exception ex) { Console.WriteLine($"  [C#] Skip {topic}: {ex.Message}"); }
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

