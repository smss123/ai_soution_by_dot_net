using LMKit.Finetuning;
using LMKit.Finetuning.Export;
using LMKit.Model;
using LMKit.TextGeneration.Chat;
using LmKitQuickStart.Training.Sources;

namespace LmKitQuickStart.Training;

/// <summary>
/// Aggregates all training sources and exports a ShareGPT dataset for Xprema LoRA fine-tuning.
/// Sources: ABP Framework docs, C# docs, ASP.NET Core docs, Sudanese dialogs.
/// </summary>
public sealed class XpremaDatasetBuilder
{
    public static readonly string DatasetPath = Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Models", "Xprema", "xprema_dataset.json"));

    private readonly IReadOnlyList<ITrainingSource> _sources;
    private readonly LM _baseModel;

    public XpremaDatasetBuilder(LM baseModel)
    {
        _baseModel = baseModel;
        _sources =
        [
            new AbpDocsTrainingSource(),
            new CSharpDocsTrainingSource(),
            new AspNetCoreDocsTrainingSource(),
            new SudaneseDialogsSource(),
            new AwniQasimDialectSource(),        // قاموس اللهجة العامية السودانية
            new SudaneseWebCrawlerSource(),      // crawler — حتى 1000 رابط من الإنترنت
        ];
    }

    /// <summary>
    /// Collects samples from all sources and exports them to a ShareGPT JSON file.
    /// Returns the path to the exported file.
    /// </summary>
    public async Task<string> BuildAsync(CancellationToken ct = default)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(DatasetPath)!);

        var allSamples = new List<ChatTrainingSample>();

        foreach (var source in _sources)
        {
            Console.WriteLine($"\n[{source.Name}] Collecting samples...");
            var histories = await source.GetSamplesAsync(_baseModel, ct);

            foreach (var history in histories)
                allSamples.Add(new ChatTrainingSample(history));

            Console.WriteLine($"  {histories.Count} samples collected.");
        }

        Console.WriteLine($"\nTotal: {allSamples.Count} training samples. Exporting...");

        var options = new DatasetBuilderOptions
        {
            Overwrite    = true,
            IndentedJson = true,
        };

        var progress = new Progress<ExportProgress>(p =>
            Console.Write($"\r  Exporting: {p.Completed}/{p.Total}  "));

        ExportResult result = await ShareGptExporter.ExportAsync(allSamples, DatasetPath, options, progress);

        Console.WriteLine($"\n  Exported {result.SamplesWritten} samples → {result.JsonPath}");
        return result.JsonPath;
    }
}
