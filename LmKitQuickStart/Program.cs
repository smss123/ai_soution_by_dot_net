using System.Text;
using LMKit.Global;
using LMKit.Model;
using LMKit.Retrieval;
using LMKit.TextGeneration;
using LMKit.TextGeneration.Chat;
using LmKitQuickStart.Rags;
using LmKitQuickStart.Training;
using System.Diagnostics;

Console.InputEncoding  = Encoding.UTF8;
Console.OutputEncoding = Encoding.UTF8;

Runtime.LogLevel   = Runtime.LMKitLogLevel.Information;
Runtime.EnableCuda = true;
Runtime.Initialize();

string basePath = GemmaModelPath();
if (!File.Exists(basePath)) { Console.WriteLine($"Base model not found:\n{basePath}"); return; }

// ── mode selection — supports --mode 1/2/3 arg for non-interactive use ──
string mode = args.FirstOrDefault(a => a.StartsWith("--mode="))?.Split('=')[1]
           ?? args.FirstOrDefault(a => a is "1" or "2" or "3" or "4")
           ?? string.Empty;

if (string.IsNullOrEmpty(mode))
{
    Console.WriteLine("Select mode:");
    Console.WriteLine("  1 — Chat");
    Console.WriteLine("  2 — Build Xprema training dataset");
    Console.WriteLine("  3 — Merge Xprema LoRA adapter → xprema.gguf");
    Console.WriteLine("  4 — Train Xprema (setup env + run fine-tuning)");
    Console.Write("Choice [1]: ");
    mode = (Console.ReadLine() ?? "1").Trim();
}

switch (mode)
{
    case "2": await RunBuildDatasetAsync(basePath);                       break;
    case "3": RunMergeAdapter(basePath);                                  break;
    case "4": await XpremaTrainer.RunAsync();                             break;
    default:  await RunChatAsync(basePath);                               break;
}

// ─────────────────────────────────────────────────────────────────
// CHAT MODE  — loads Xprema if merged model exists, else base Gemma
// ─────────────────────────────────────────────────────────────────
static async Task RunChatAsync(string basePath)
{
    using LM chatModel = XpremaFineTuner.Load(basePath);

    AbpDocsKnowledgeBase?   abpKb  = null;
    RagAbpDocs?             abpRag = null;
    RagSamerCv?             cvRag  = null;
    SingleTurnConversation? cvChat = null;

    string modelName = XpremaFineTuner.MergedExists ? "Xprema" : "Gemma";
    Console.WriteLine($"[{modelName}] Ask anything — AI routes to the right knowledge base.");
    Console.WriteLine("  ABP Framework docs  |  Samer CV  |  'quit' to exit\n");

    while (true)
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.Write("Question: ");
        Console.ResetColor();

        string? query = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(query) || query.Equals("quit", StringComparison.OrdinalIgnoreCase))
            break;

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.Write("  Routing... ");
        Console.ResetColor();

        RagTarget target = ClassifyQuery(query, chatModel);

        if (target == RagTarget.Unknown)
        {
            Console.WriteLine("[no matching knowledge base]\n");
            continue;
        }

        if (target == RagTarget.Abp)
        {
            PrintLabel("ABP Docs");

            if (abpKb is null)
            {
                abpKb  = new AbpDocsKnowledgeBase();
                await abpKb.EnsureIndexedAsync();
                abpRag = new RagAbpDocs(abpKb);
            }

            using RagChat ragChat = abpRag!.CreateRagChat(chatModel);
            ragChat.AfterTextCompletion += (_, e) =>
            {
                if (e.SegmentType == TextSegmentType.UserVisible) Console.Write(e.Text);
            };
            PrintAnswerPrompt();

            var result = await Task.Run(() => ragChat.Submit(query));
            PrintStats(result.Response.GeneratedTokenCount, result.Response.TokenGenerationRate);
        }
        else
        {
            PrintLabel("Samer CV");

            if (cvRag is null)
            {
                cvRag  = new RagSamerCv();
                cvRag.EnsureIndexed();
                cvChat = new SingleTurnConversation(chatModel)
                {
                    SystemPrompt            = "Answer using only the provided context. If not found, say so.",
                    MaximumCompletionTokens = 512
                };
                cvChat.AfterTextCompletion += (_, e) =>
                {
                    if (e.SegmentType == TextSegmentType.UserVisible) Console.Write(e.Text);
                };
            }

            var matches = cvRag.FindMatches(query);
            if (matches.Count == 0) { Console.WriteLine("  [No relevant passages found.]\n"); continue; }

            Console.ForegroundColor = ConsoleColor.DarkGray;
            foreach (var m in matches)
                Console.WriteLine($"  [{m.SectionIdentifier}] {m.Similarity:F3}");
            Console.ResetColor();

            PrintAnswerPrompt();
            var result = cvRag.Answer(query, matches, cvChat!);
            PrintStats(result.GeneratedTokenCount, result.TokenGenerationRate);
        }
    }

    cvRag?.Dispose();
    abpKb?.Dispose();
}

// ─────────────────────────────────────────────────────────────────
// BUILD DATASET MODE
// ─────────────────────────────────────────────────────────────────
static async Task RunBuildDatasetAsync(string basePath)
{
    Console.WriteLine("\nLoading base model for dataset building...");
    using LM baseModel = new LM(basePath,
        deviceConfiguration: new LM.DeviceConfiguration { GpuLayerCount = 40 },
        loadingProgress: p => { Console.Write($"\r  Loading {p * 100:F0}%  "); return true; });
    Console.WriteLine(" Done.\n");

    Console.WriteLine("=== Building Xprema training dataset ===");
    var builder = new XpremaDatasetBuilder(baseModel);
    string datasetPath = await builder.BuildAsync();

    Console.WriteLine($"""

Dataset ready: {datasetPath}

Fine-tune externally, then place the adapter at:
  {XpremaFineTuner.AdapterPath}

Finally run mode 3 to merge → xprema.gguf, then mode 1 to chat.
""");
}

// ─────────────────────────────────────────────────────────────────
// MERGE ADAPTER MODE
// ─────────────────────────────────────────────────────────────────
static void RunMergeAdapter(string basePath)
{
    Console.WriteLine();
    XpremaFineTuner.MergeAdapter(basePath);
    Console.WriteLine("\nRun mode 1 to chat with Xprema.");
}

// ── shared helpers ─────────────────────────────────────────────────

static string GemmaModelPath() => Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
    ".lmstudio", "models",
    "lmstudio-community", "gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf");

static RagTarget ClassifyQuery(string query, LM model)
{
    const string prompt =
        "You are a query router. Reply with exactly one word: ABP, CV, or NONE.\n" +
        "  ABP  = ABP Framework, modules, DDD, EF Core, Blazor, Angular, tenancy, permissions, CLI\n" +
        "  CV   = Samer's resume, experience, skills, education, projects, certifications\n" +
        "  NONE = neither";

    var classifier = new SingleTurnConversation(model)
    {
        SystemPrompt            = prompt,
        MaximumCompletionTokens = 5,
    };

    string label = classifier.SubmitAsync(query).GetAwaiter().GetResult()
                              .Completion.Trim().ToUpperInvariant();

    if (label.StartsWith("ABP")) return RagTarget.Abp;
    if (label.StartsWith("CV"))  return RagTarget.Cv;
    return RagTarget.Unknown;
}

static void PrintLabel(string label)
{
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine($"→ {label}");
    Console.ResetColor();
}

static void PrintAnswerPrompt()
{
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.Write("Answer: ");
    Console.ResetColor();
}

static void PrintStats(int tokens, double rate) =>
    Console.WriteLine($"\n  [{tokens} tokens, {rate:F1} tok/s]\n");

enum RagTarget { Unknown, Abp, Cv }
