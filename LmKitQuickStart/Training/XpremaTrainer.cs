using System.Diagnostics;
using System.Runtime.InteropServices;

namespace LmKitQuickStart.Training;

/// <summary>
/// Launches the Unsloth Python training pipeline from within the .NET application.
/// Handles environment setup, training, and reports live progress to the console.
/// </summary>
public static class XpremaTrainer
{
    private static readonly string TrainingDir = Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Training"));

    private static readonly string EnvDir      = Path.Combine(TrainingDir, "xprema_env");
    private static readonly string TrainScript = Path.Combine(TrainingDir, "unsloth_train.py");

    private static string PythonExe => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        ? Path.Combine(EnvDir, "Scripts", "python.exe")
        : Path.Combine(EnvDir, "bin", "python");

    private static string PipExe => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        ? Path.Combine(EnvDir, "Scripts", "pip.exe")
        : Path.Combine(EnvDir, "bin", "pip");

    public static async Task RunAsync(CancellationToken ct = default)
    {
        Console.WriteLine("\n=== Xprema Training Pipeline ===\n");

        if (!await EnsureEnvironmentAsync(ct)) return;
        if (!await EnsureDependenciesAsync(ct)) return;
        await RunTrainingAsync(ct);
    }

    // ── Step 1: ensure Python venv exists ────────────────────────────

    private static async Task<bool> EnsureEnvironmentAsync(CancellationToken ct)
    {
        if (File.Exists(PythonExe))
        {
            Console.WriteLine("[env] Python environment already exists.");
            return true;
        }

        Console.WriteLine("[env] Creating Python 3.11 virtual environment...");

        int exit = await RunCommandAsync("uv", $"venv \"{EnvDir}\" --python 3.11", TrainingDir, ct);
        if (exit != 0)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("[env] FAILED. Make sure 'uv' is installed: winget install astral-sh.uv");
            Console.ResetColor();
            return false;
        }

        // Bootstrap pip (uv venv doesn't include pip by default)
        await RunCommandAsync(PythonExe, "-m ensurepip --upgrade", TrainingDir, ct);
        await RunCommandAsync(PythonExe, "-m pip install --upgrade pip", TrainingDir, ct);

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("[env] Environment ready.");
        Console.ResetColor();
        return true;
    }

    // ── Step 2: install dependencies ─────────────────────────────────

    private static async Task<bool> EnsureDependenciesAsync(CancellationToken ct)
    {
        // Quick check: if torch is importable, dependencies are already installed
        int check = await RunCommandAsync(PythonExe,
            "-c \"import torch; import unsloth; import trl\"", TrainingDir, ct, silent: true);

        if (check == 0)
        {
            Console.WriteLine("[deps] All dependencies already installed.");
            return true;
        }

        Console.WriteLine("[deps] Installing PyTorch (CUDA 12.1) — this may take a few minutes...");
        int exit = await RunCommandAsync(PipExe,
            "install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
            TrainingDir, ct);

        if (exit != 0)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("[deps] PyTorch installation failed.");
            Console.ResetColor();
            return false;
        }

        Console.WriteLine("[deps] Installing Unsloth + TRL + datasets...");
        exit = await RunCommandAsync(PipExe,
            "install \"unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git\" " +
            "\"trl>=0.15,<0.19\" \"transformers>=4.49,<4.57\" \"tokenizers>=0.21,<0.22\" " +
            "\"datasets>=3.0,<3.7\" \"huggingface_hub>=0.27,<0.31\" \"accelerate>=1.2,<1.9\" " +
            "\"peft>=0.14,<0.19\" bitsandbytes numpy",
            TrainingDir, ct);

        if (exit != 0)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("[deps] Dependency installation failed.");
            Console.ResetColor();
            return false;
        }

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("[deps] All dependencies installed.");
        Console.ResetColor();
        return true;
    }

    // ── Step 3: run training script ───────────────────────────────────

    private static async Task RunTrainingAsync(CancellationToken ct)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n[train] Starting Xprema LoRA fine-tuning...");
        Console.ResetColor();

        var sw = Stopwatch.StartNew();
        int exit = await RunCommandAsync(PythonExe, $"\"{TrainScript}\"", TrainingDir, ct);
        sw.Stop();

        if (exit == 0)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n[train] Training complete in {sw.Elapsed:hh\\:mm\\:ss}.");
            Console.ResetColor();
            Console.WriteLine("LoRA adapter saved in HuggingFace format (see script output above).");
            Console.WriteLine($"Convert it to GGUF with llama.cpp's convert_lora_to_gguf.py, saving to:\n  {XpremaFineTuner.AdapterPath}");
            Console.WriteLine("Then run mode 3 to merge → xprema.gguf, then mode 1 to chat.");
        }
        else
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n[train] Training failed (exit code {exit}).");
            Console.ResetColor();
        }
    }

    // ── Process runner ────────────────────────────────────────────────

    private static async Task<int> RunCommandAsync(
        string exe, string args, string workDir, CancellationToken ct, bool silent = false)
    {
        var psi = new ProcessStartInfo(exe, args)
        {
            WorkingDirectory       = workDir,
            UseShellExecute        = false,
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
        };

        using var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };

        if (!silent)
        {
            proc.OutputDataReceived += (_, e) => { if (e.Data != null) Console.WriteLine(e.Data); };
            proc.ErrorDataReceived  += (_, e) => { if (e.Data != null) Console.WriteLine(e.Data); };
        }

        proc.Start();

        if (!silent)
        {
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
        }

        await proc.WaitForExitAsync(ct);
        return proc.ExitCode;
    }
}
