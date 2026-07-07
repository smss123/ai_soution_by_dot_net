using System.Text;
using LMKit.Global;
using LMKit.Model;
using LMKit.Retrieval;
using LMKit.TextGeneration;
using LMKit.TextGeneration.Chat;
using LMKit.TextGeneration.Sampling;
using LmKitQuickStart.Rags;
using LmKitQuickStart.Training;
using System.Diagnostics;

Console.InputEncoding  = Encoding.UTF8;
Console.OutputEncoding = Encoding.UTF8;

Runtime.LogLevel   = Runtime.LMKitLogLevel.Information;
Runtime.EnableCuda = true;
Runtime.Initialize();

string basePath = BaseModelPath();

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

// Mode 4 (Unsloth training) is pure Python and doesn't touch the local GGUF.
if (mode != "4" && !File.Exists(basePath))
{
    Console.WriteLine($"Base model not found:\n{basePath}");
    return;
}

switch (mode)
{
    case "2": await RunBuildDatasetAsync(basePath);                       break;
    case "3": RunMergeAdapter(basePath);                                  break;
    case "4": await XpremaTrainer.RunAsync();                             break;
    default:  await RunChatAsync(basePath);                               break;
}

// ─────────────────────────────────────────────────────────────────
// CHAT MODE  — loads Xprema if merged model exists, else the base model
// ─────────────────────────────────────────────────────────────────
static async Task RunChatAsync(string basePath)
{
    using LM chatModel = XpremaFineTuner.Load(basePath);

    // Xprema is a 0.5B model — too weak to reliably follow the routing
    // instruction, so a separate larger model handles query classification.
    Console.WriteLine("Loading router model...");
    using LM routerModel = new LM(RouterModelPath(),
        deviceConfiguration: new LM.DeviceConfiguration { GpuLayerCount = 40 },
        loadingProgress: p => { Console.Write($"\r  Loading {p * 100:F0}%  "); return true; });
    Console.WriteLine(" Done.\n");

    AbpDocsKnowledgeBase?   abpKb  = null;
    RagAbpDocs?             abpRag = null;
    RagSamerCv?             cvRag  = null;
    SingleTurnConversation? cvChat = null;

    SingleTurnConversation? modelChat = null;

    string modelName = XpremaFineTuner.MergedExists ? "Xprema" : "Base";
    Console.WriteLine($"[{modelName}] Ask anything — AI routes to the right knowledge base.");
    Console.WriteLine("  ABP Framework docs  |  Samer CV  |  Questions about the model  |  'quit' to exit\n");

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

        RagTarget target = ClassifyQuery(query, routerModel);

        if (target == RagTarget.Unknown)
        {
            Console.WriteLine("[no matching knowledge base]\n");
            continue;
        }

        try
        {
        if (target == RagTarget.Model)
        {
            PrintLabel("About Xprema");

            if (modelChat is null)
            {
                // Xprema (0.5B) is too weak to answer coherently about itself,
                // so the larger router model generates this answer instead.
                modelChat = new SingleTurnConversation(routerModel)
                {
                    SystemPrompt = "You are Xprema, a small AI model fine-tuned from Qwen2.5-0.5B-Instruct " +
                                   "for Direct Aid Society. Answer questions about yourself briefly and " +
                                   "honestly. Always reply in the same language the user wrote in.",
                    MaximumCompletionTokens = 256,
                    SamplingMode      = new RandomSampling { Temperature = 0.7f, TopP = 0.9f, TopK = 40 },
                    RepetitionPenalty = { TokenCount = 256, RepeatPenalty = 1.3f, PresencePenalty = 0.6f },
                };
                modelChat.AfterTextCompletion += (_, e) =>
                {
                    if (e.SegmentType == TextSegmentType.UserVisible) Console.Write(e.Text);
                };
            }

            PrintAnswerPrompt();
            var result = await Task.Run(() => modelChat.Submit(query));
            PrintStats(result.GeneratedTokenCount, result.TokenGenerationRate);
        }
        else if (target == RagTarget.Abp)
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
                    SystemPrompt            = "Answer using only the provided context. If not found, say so. " +
                                               "Always reply in the same language the user wrote in.",
                    MaximumCompletionTokens = 512,
                    SamplingMode      = new RandomSampling { Temperature = 0.7f, TopP = 0.9f, TopK = 40 },
                    RepetitionPenalty = { TokenCount = 256, RepeatPenalty = 1.3f, PresencePenalty = 0.6f },
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
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n  [Error answering query: {ex.Message}]\n");
            Console.ResetColor();
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

static string BaseModelPath() => Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
    ".lmstudio", "models",
    "lmstudio-community", "Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-fp16.gguf");

static string RouterModelPath() => Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
    ".lmstudio", "models",
    "lmstudio-community", "Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf");

static RagTarget ClassifyQuery(string query, LM model)
{
    const string prompt =
        "You are a query router. The user may write in any language, including Arabic. " +
        "Reply with exactly one word: ABP, CV, MODEL, or NONE.\n" +
        "  ABP   = ABP Framework, modules, DDD, EF Core, Blazor, Angular, tenancy, permissions, CLI\n" +
        "  CV    = Samer's resume, experience, skills, education, projects, certifications\n" +
        "  MODEL = the question is about the AI assistant itself — its name, who made/trained it, " +
        "what model it is, or its abilities\n" +
        "  NONE  = none of the above\n" +
        "Examples:\n" +
        "  \"what is your name\" -> MODEL\n" +
        "  \"who trained you\" -> MODEL\n" +
        "  \"ما اسمك\" -> MODEL\n" +
        "  \"من دربك\" -> MODEL\n" +
        "  \"what is dependency injection in abp\" -> ABP\n" +
        "  \"what are samer's skills\" -> CV\n" +
        "  \"who is samer\" -> CV\n" +
        "  \"من هو سامر\" -> CV";

    var classifier = new SingleTurnConversation(model)
    {
        SystemPrompt            = prompt,
        MaximumCompletionTokens = 5,
    };

    string label = classifier.SubmitAsync(query).GetAwaiter().GetResult()
                              .Completion.Trim().ToUpperInvariant();

    if (label.StartsWith("ABP"))   return RagTarget.Abp;
    if (label.StartsWith("CV"))    return RagTarget.Cv;
    if (label.StartsWith("MODEL")) return RagTarget.Model;
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

enum RagTarget { Unknown, Abp, Cv, Model }
