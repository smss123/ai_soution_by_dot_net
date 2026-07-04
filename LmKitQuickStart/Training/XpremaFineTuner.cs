using LMKit.Finetuning;
using LMKit.Model;

namespace LmKitQuickStart.Training;

/// <summary>
/// Manages the Xprema model lifecycle.
///
/// Fine-tuning workflow:
///   1. Run mode 2  → builds and exports ShareGPT dataset to Models/Xprema/
///   2. Train externally (llama.cpp / Unsloth) using the exported JSON
///   3. Place the LoRA adapter at:  Models/Xprema/xprema-adapter.gguf
///   4. Run mode 2 again → selects "Merge adapter" to produce xprema.gguf
///   5. Run mode 1  → Xprema loads as the default chat model
///
/// The merged model (xprema.gguf) is a standalone GGUF — no base model needed at runtime.
/// </summary>
public static class XpremaFineTuner
{
    private static readonly string XpremaFolder = Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Models", "Xprema"));

    public static readonly string AdapterPath = Path.Combine(XpremaFolder, "xprema-adapter.gguf");
    public static readonly string MergedPath  = Path.Combine(XpremaFolder, "xprema.gguf");

    public static bool AdapterExists => File.Exists(AdapterPath);
    public static bool MergedExists  => File.Exists(MergedPath);

    /// <summary>
    /// Merges the LoRA adapter into the base model and saves xprema.gguf.
    /// Only needed once after external LoRA training is complete.
    /// </summary>
    public static void MergeAdapter(string basePath)
    {
        if (!File.Exists(AdapterPath))
        {
            Console.WriteLine($"Adapter not found: {AdapterPath}");
            Console.WriteLine("Train first, then place the adapter file there.");
            return;
        }

        Directory.CreateDirectory(XpremaFolder);

        Console.WriteLine("Merging LoRA adapter into base model...");
        Console.WriteLine($"  Base    : {basePath}");
        Console.WriteLine($"  Adapter : {AdapterPath}");
        Console.WriteLine($"  Output  : {MergedPath}");

        var merger = new LoraMerger(basePath);
        merger.AddLoraAdapter(AdapterPath, scale: 1.0f);
        merger.Merge(MergedPath);

        Console.WriteLine("Merge complete. Xprema model saved.");
    }

    /// <summary>
    /// Loads the Xprema merged model if it exists, otherwise loads base Gemma.
    /// </summary>
    public static LM Load(string basePath)
    {
        string modelToLoad = MergedExists ? MergedPath : basePath;
        string label       = MergedExists ? "Xprema"   : "base Gemma (Xprema not trained yet)";

        Console.WriteLine($"Loading {label}...");

        var model = new LM(modelToLoad,
            deviceConfiguration: new LM.DeviceConfiguration { GpuLayerCount = 40 },
            loadingProgress: p => { Console.Write($"\r  Loading {p * 100:F0}%  "); return true; });

        Console.WriteLine(" Done.\n");

        if (!MergedExists)
            PrintTrainingInstructions();

        return model;
    }

    private static void PrintTrainingInstructions()
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine("""
  To train Xprema:
    1. Run mode 2 → "Build dataset" to generate Models/Xprema/xprema_dataset.json
    2. Fine-tune with llama.cpp:
         llama-finetune --model base.gguf --train-data xprema_dataset.json --lora-out xprema-adapter.gguf
       Or with Unsloth (Python):
         python train.py --dataset xprema_dataset.json --output xprema-adapter.gguf
    3. Place adapter at: Models/Xprema/xprema-adapter.gguf
    4. Run mode 2 → "Merge adapter" to produce xprema.gguf
    5. Run mode 1 to chat with Xprema
""");
        Console.ResetColor();
    }
}
