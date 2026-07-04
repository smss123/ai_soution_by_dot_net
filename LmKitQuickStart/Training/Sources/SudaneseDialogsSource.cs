using System.Text.Json;
using LMKit.Model;
using LMKit.TextGeneration.Chat;

namespace LmKitQuickStart.Training.Sources;

/// <summary>
/// Loads Sudanese dialect training dialogs from TrainingData/sudanese_dialogs.json.
/// Add or edit entries in that file — no code changes needed.
/// </summary>
public sealed class SudaneseDialogsSource : ITrainingSource
{
    public string Name => "Sudanese Dialogs";

    private static readonly string DataFile = Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "TrainingData", "sudanese_dialogs.json"));

    private const string SystemPrompt =
        "أنت Xprema، مساعد ذكاء اصطناعي متخصص يفهم اللهجة السودانية ويجيب بشكل طبيعي.";

    public Task<IReadOnlyList<ChatHistory>> GetSamplesAsync(LM baseModel, CancellationToken ct = default)
    {
        var dialogs = LoadDialogs();
        var samples = new List<ChatHistory>(dialogs.Count);

        foreach (var dialog in dialogs)
        {
            ct.ThrowIfCancellationRequested();
            var history = new ChatHistory(baseModel);
            history.AddMessage(AuthorRole.System,    SystemPrompt);
            history.AddMessage(AuthorRole.User,      dialog.User);
            history.AddMessage(AuthorRole.Assistant, dialog.Assistant);
            samples.Add(history);
        }

        Console.WriteLine($"  Loaded {samples.Count} dialog samples from file.");
        return Task.FromResult<IReadOnlyList<ChatHistory>>(samples);
    }

    private static List<DialogEntry> LoadDialogs()
    {
        if (!File.Exists(DataFile))
        {
            Console.WriteLine($"  [Warning] {DataFile} not found — skipping Sudanese dialogs.");
            return [];
        }

        string json = File.ReadAllText(DataFile);
        return JsonSerializer.Deserialize<List<DialogEntry>>(json,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? [];
    }

    private sealed record DialogEntry(string User, string Assistant);
}
