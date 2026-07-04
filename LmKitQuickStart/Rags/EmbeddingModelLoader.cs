using LMKit.Model;

namespace LmKitQuickStart.Rags;

/// <summary>
/// Loads the shared embedding model from the project's Models/ folder.
/// Falls back to the LM-Kit download cache, then downloads if necessary.
/// After the first load the file always lives in Models/ for future runs.
/// </summary>
public static class EmbeddingModelLoader
{
    private const string ModelId = "embeddinggemma-300m";

    public static readonly string ModelsFolder = Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Models"));

    private static readonly string[] DefaultCachePaths =
    [
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "models", "lm-kit"),
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "LM-Kit", "Models"),
    ];

    public static LM Load()
    {
        Directory.CreateDirectory(ModelsFolder);

        string? localFile = FindGguf(ModelsFolder);
        if (localFile is not null)
        {
            Console.WriteLine("Loading embedding model from Models/ folder...");
            return LoadFromPath(localFile);
        }

        foreach (string cache in DefaultCachePaths)
        {
            string? cached = FindGguf(cache);
            if (cached is null) continue;

            string destination = Path.Combine(ModelsFolder, Path.GetFileName(cached));
            Console.WriteLine("Moving embedding model to Models/ folder...");
            try   { File.Move(cached, destination); return LoadFromPath(destination); }
            catch { return LoadFromPath(cached); }
        }

        Console.WriteLine("Downloading embedding model into Models/ folder...");
        LM downloaded = LM.LoadFromModelID(ModelId,
            downloadingProgress: (_, total, read) =>
            {
                if (total.HasValue) Console.Write($"\r  {(double)read / total.Value * 100:F1}%  ");
                return true;
            },
            loadingProgress: p => { Console.Write($"\r  Loading {p * 100:F0}%  "); return true; });

        TryCopyToModelsFolder();
        return downloaded;
    }

    private static string? FindGguf(string folder) =>
        Directory.Exists(folder)
            ? Directory.EnumerateFiles(folder, "*.gguf", SearchOption.AllDirectories)
                       .FirstOrDefault(f => f.Contains("embedding", StringComparison.OrdinalIgnoreCase)
                                         && f.Contains("gemma",     StringComparison.OrdinalIgnoreCase))
            : null;

    private static LM LoadFromPath(string path) =>
        new(path, loadingProgress: p => { Console.Write($"\r  Loading {p * 100:F0}%  "); return true; });

    private static void TryCopyToModelsFolder()
    {
        foreach (string cache in DefaultCachePaths)
        {
            string? file = FindGguf(cache);
            if (file is null) continue;
            try { File.Copy(file, Path.Combine(ModelsFolder, Path.GetFileName(file)), overwrite: false); }
            catch { /* best-effort */ }
            break;
        }
    }
}
