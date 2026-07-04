using LMKit.Model;
using LMKit.TextGeneration.Chat;

namespace LmKitQuickStart.Training;

/// <summary>
/// Contract for a single domain training data source.
/// Each source produces ChatHistory samples ready for LoRA fine-tuning.
/// </summary>
public interface ITrainingSource
{
    string Name { get; }

    Task<IReadOnlyList<ChatHistory>> GetSamplesAsync(LM baseModel, CancellationToken ct = default);
}
