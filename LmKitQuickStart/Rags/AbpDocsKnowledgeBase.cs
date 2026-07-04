using System.Text;
using HtmlAgilityPack;
using LMKit.Data;
using LMKit.Model;
using LMKit.Retrieval;

namespace LmKitQuickStart.Rags;

/// <summary>
/// Manages the persisted ABP Framework documentation DataSource.
/// Crawls official doc pages, indexes them, and exposes the DataSource
/// for any RAG engine to attach to.
///
/// Usage:
///   using var kb = new AbpDocsKnowledgeBase();
///   await kb.EnsureIndexedAsync();
///   var rag = new RagAbpDocs(kb);
/// </summary>
public sealed class AbpDocsKnowledgeBase : IDisposable
{
    private static readonly string IndexPath =
        Path.Combine(AppContext.BaseDirectory, "abp_docs.dat");

    private const string DataSourceName = "AbpDocs";
    private const string BaseUrl        = "https://abp.io";
    private const int    CrawlDelayMs   = 300;

    public static readonly (string Section, string Url)[] DocPages =
    [
        ("get-started",                    $"{BaseUrl}/docs/latest/get-started"),
        ("get-started/single-layer",       $"{BaseUrl}/docs/latest/get-started/single-layer-web-application"),
        ("get-started/layered",            $"{BaseUrl}/docs/latest/get-started/layered-web-application"),
        ("get-started/microservice",       $"{BaseUrl}/docs/latest/get-started/microservice"),
        ("get-started/empty-aspnet",       $"{BaseUrl}/docs/latest/get-started/empty-aspnet-core-application"),
        ("get-started/console",            $"{BaseUrl}/docs/latest/get-started/console"),
        ("get-started/pre-requirements",   $"{BaseUrl}/docs/latest/get-started/pre-requirements"),
        ("tutorials/todo",                 $"{BaseUrl}/docs/latest/tutorials/todo"),
        ("tutorials/book-store",           $"{BaseUrl}/docs/latest/tutorials/book-store"),
        ("cli",                            $"{BaseUrl}/docs/latest/cli"),
        ("fundamentals/application-startup",  $"{BaseUrl}/docs/latest/framework/fundamentals/application-startup"),
        ("fundamentals/authorization",        $"{BaseUrl}/docs/latest/framework/fundamentals/authorization"),
        ("fundamentals/caching",              $"{BaseUrl}/docs/latest/framework/fundamentals/caching"),
        ("fundamentals/configuration",        $"{BaseUrl}/docs/latest/framework/fundamentals/configuration"),
        ("fundamentals/connection-strings",   $"{BaseUrl}/docs/latest/framework/fundamentals/connection-strings"),
        ("fundamentals/dependency-injection", $"{BaseUrl}/docs/latest/framework/fundamentals/dependency-injection"),
        ("fundamentals/exception-handling",   $"{BaseUrl}/docs/latest/framework/fundamentals/exception-handling"),
        ("fundamentals/localization",         $"{BaseUrl}/docs/latest/framework/fundamentals/localization"),
        ("fundamentals/logging",              $"{BaseUrl}/docs/latest/framework/fundamentals/logging"),
        ("fundamentals/options",              $"{BaseUrl}/docs/latest/framework/fundamentals/options"),
        ("fundamentals/validation",           $"{BaseUrl}/docs/latest/framework/fundamentals/validation"),
        ("infra/audit-logging",              $"{BaseUrl}/docs/latest/framework/infrastructure/audit-logging"),
        ("infra/background-jobs",            $"{BaseUrl}/docs/latest/framework/infrastructure/background-jobs"),
        ("infra/background-workers",         $"{BaseUrl}/docs/latest/framework/infrastructure/background-workers"),
        ("infra/blob-storing",               $"{BaseUrl}/docs/latest/framework/infrastructure/blob-storing"),
        ("infra/caching/redis",              $"{BaseUrl}/docs/latest/framework/fundamentals/redis-cache"),
        ("infra/current-user",               $"{BaseUrl}/docs/latest/framework/infrastructure/current-user"),
        ("infra/data-filtering",             $"{BaseUrl}/docs/latest/framework/infrastructure/data-filtering"),
        ("infra/data-seeding",               $"{BaseUrl}/docs/latest/framework/infrastructure/data-seeding"),
        ("infra/distributed-locking",        $"{BaseUrl}/docs/latest/framework/infrastructure/distributed-locking"),
        ("infra/emailing",                   $"{BaseUrl}/docs/latest/framework/infrastructure/emailing"),
        ("infra/event-bus",                  $"{BaseUrl}/docs/latest/framework/infrastructure/event-bus"),
        ("infra/event-bus/local",            $"{BaseUrl}/docs/latest/framework/infrastructure/event-bus/local"),
        ("infra/event-bus/distributed",      $"{BaseUrl}/docs/latest/framework/infrastructure/event-bus/distributed"),
        ("infra/features",                   $"{BaseUrl}/docs/latest/framework/infrastructure/features"),
        ("infra/settings",                   $"{BaseUrl}/docs/latest/framework/infrastructure/settings"),
        ("infra/virtual-file-system",        $"{BaseUrl}/docs/latest/framework/infrastructure/virtual-file-system"),
        ("infra/multi-tenancy",              $"{BaseUrl}/docs/latest/framework/architecture/multi-tenancy"),
        ("architecture/modularity",          $"{BaseUrl}/docs/latest/framework/architecture/modularity/basics"),
        ("architecture/ddd",                 $"{BaseUrl}/docs/latest/framework/architecture/domain-driven-design"),
        ("architecture/ddd/entities",        $"{BaseUrl}/docs/latest/framework/architecture/domain-driven-design/entities"),
        ("architecture/ddd/repositories",    $"{BaseUrl}/docs/latest/framework/architecture/domain-driven-design/repositories"),
        ("architecture/ddd/domain-services", $"{BaseUrl}/docs/latest/framework/architecture/domain-driven-design/domain-services"),
        ("architecture/ddd/app-services",    $"{BaseUrl}/docs/latest/framework/architecture/domain-driven-design/application-services"),
        ("architecture/ddd/dtos",            $"{BaseUrl}/docs/latest/framework/architecture/domain-driven-design/data-transfer-objects"),
        ("architecture/ddd/unit-of-work",    $"{BaseUrl}/docs/latest/framework/architecture/domain-driven-design/unit-of-work"),
        ("architecture/microservices",       $"{BaseUrl}/docs/latest/framework/architecture/microservices"),
        ("api/auto-controllers",             $"{BaseUrl}/docs/latest/framework/api-development/auto-controllers"),
        ("api/dynamic-csharp-clients",       $"{BaseUrl}/docs/latest/framework/api-development/dynamic-csharp-clients"),
        ("api/swagger",                      $"{BaseUrl}/docs/latest/framework/api-development/swagger"),
        ("api/versioning",                   $"{BaseUrl}/docs/latest/framework/api-development/versioning"),
        ("data/efcore",                      $"{BaseUrl}/docs/latest/framework/data/entity-framework-core"),
        ("data/efcore/migrations",           $"{BaseUrl}/docs/latest/framework/data/entity-framework-core/migrations"),
        ("data/mongodb",                     $"{BaseUrl}/docs/latest/framework/data/mongodb"),
        ("ui/mvc/overall",                   $"{BaseUrl}/docs/latest/framework/ui/mvc-razor-pages/overall"),
        ("ui/mvc/navigation-menu",           $"{BaseUrl}/docs/latest/framework/ui/mvc-razor-pages/navigation-menu"),
        ("ui/mvc/theming",                   $"{BaseUrl}/docs/latest/framework/ui/mvc-razor-pages/theming"),
        ("ui/mvc/bundling",                  $"{BaseUrl}/docs/latest/framework/ui/mvc-razor-pages/bundling-minification"),
        ("ui/blazor/overall",                $"{BaseUrl}/docs/latest/framework/ui/blazor/overall"),
        ("ui/blazor/theming",                $"{BaseUrl}/docs/latest/framework/ui/blazor/theming"),
        ("ui/angular/overview",              $"{BaseUrl}/docs/latest/framework/ui/angular/overview"),
        ("ui/angular/authorization",         $"{BaseUrl}/docs/latest/framework/ui/angular/authorization"),
        ("ui/angular/localization",          $"{BaseUrl}/docs/latest/framework/ui/angular/localization"),
        ("ui/angular/theming",               $"{BaseUrl}/docs/latest/framework/ui/angular/theming"),
        ("modules/account",                  $"{BaseUrl}/docs/latest/modules/account"),
        ("modules/audit-logging",            $"{BaseUrl}/docs/latest/modules/audit-logging"),
        ("modules/background-jobs",          $"{BaseUrl}/docs/latest/modules/background-jobs"),
        ("modules/identity",                 $"{BaseUrl}/docs/latest/modules/identity"),
        ("modules/permission-management",    $"{BaseUrl}/docs/latest/modules/permission-management"),
        ("modules/setting-management",       $"{BaseUrl}/docs/latest/modules/setting-management"),
        ("modules/tenant-management",        $"{BaseUrl}/docs/latest/modules/tenant-management"),
        ("modules/feature-management",       $"{BaseUrl}/docs/latest/modules/feature-management"),
        ("modules/openiddict",               $"{BaseUrl}/docs/latest/modules/openiddict"),
        ("modules/cms-kit",                  $"{BaseUrl}/docs/latest/modules/cms-kit"),
        ("templates/single-layer",           $"{BaseUrl}/docs/latest/solution-templates/single-layer-web-application"),
        ("templates/layered",                $"{BaseUrl}/docs/latest/solution-templates/layered-web-application"),
        ("templates/microservice",           $"{BaseUrl}/docs/latest/solution-templates/microservice"),
        ("testing/overall",                  $"{BaseUrl}/docs/latest/testing/overall"),
        ("testing/unit-tests",               $"{BaseUrl}/docs/latest/testing/unit-tests"),
        ("testing/integration-tests",        $"{BaseUrl}/docs/latest/testing/integration-tests"),
        ("deployment",                       $"{BaseUrl}/docs/latest/deployment"),
        ("deployment/production",            $"{BaseUrl}/docs/latest/deployment/configuring-production"),
        ("deployment/clustered",             $"{BaseUrl}/docs/latest/deployment/clustered-environment"),
    ];

    public LM         EmbeddingModel { get; }
    public DataSource DataSource     { get; }

    public AbpDocsKnowledgeBase()
    {
        EmbeddingModel = EmbeddingModelLoader.Load();
        Console.WriteLine(" Done.\n");
        DataSource = OpenOrCreate(IndexPath, DataSourceName, EmbeddingModel);
    }

    public async Task EnsureIndexedAsync()
    {
        using var http   = CreateHttpClient();
        int indexed = 0, skipped = 0;

        foreach (var (section, url) in DocPages)
        {
            if (DataSource.HasSection(section)) { skipped++; continue; }

            try
            {
                Console.Write($"  Fetching {section}... ");
                string html = await http.GetStringAsync(url);
                string text = HtmlToText(html);

                if (string.IsNullOrWhiteSpace(text)) { Console.WriteLine("(empty)"); continue; }

                IndexText(text, section);
                Console.WriteLine("OK");
                indexed++;

                await Task.Delay(CrawlDelayMs);
            }
            catch (Exception ex) { Console.WriteLine($"ERROR: {ex.Message}"); }
        }

        Console.WriteLine($"\n  Indexed {indexed} new, {skipped} cached. Total: {DataSource.Sections.Count()} section(s).\n");
    }

    public async Task UpdateSectionsAsync(params string[] sectionNames)
    {
        using var http   = CreateHttpClient();
        var urlMap = DocPages.ToDictionary(p => p.Section, p => p.Url);

        foreach (string section in sectionNames)
        {
            if (!urlMap.TryGetValue(section, out string? url))
            {
                Console.WriteLine($"  Unknown section: {section}");
                continue;
            }

            try
            {
                Console.Write($"  Updating {section}... ");
                string text = HtmlToText(await http.GetStringAsync(url));
                if (!string.IsNullOrWhiteSpace(text)) IndexText(text, section);
                Console.WriteLine("OK");
                await Task.Delay(CrawlDelayMs);
            }
            catch (Exception ex) { Console.WriteLine($"ERROR: {ex.Message}"); }
        }
    }

    private void IndexText(string text, string section)
    {
        var tempEngine = new RagEngine(EmbeddingModel);
        tempEngine.AddDataSource(DataSource);
        tempEngine.DefaultIChunking = new TextChunking
        {
            MaxChunkSize  = RagEngineBase.ChunkSize,
            MaxOverlapSize = RagEngineBase.ChunkOverlap
        };
        tempEngine.ImportText(text, DataSourceName, section);
    }

    private static DataSource OpenOrCreate(string path, string name, LM model)
    {
        if (File.Exists(path))
        {
            try   { return DataSource.LoadFromFile(path, readOnly: false); }
            catch { Console.WriteLine("  [Index corrupt — rebuilding...]"); File.Delete(path); }
        }
        return DataSource.CreateFileDataSource(path, name, model);
    }

    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("Mozilla/5.0 (compatible; AbpDocsRag/1.0)");
        return client;
    }

    private static string HtmlToText(string html)
    {
        var doc = new HtmlDocument();
        doc.LoadHtml(html);

        foreach (string tag in (string[])["script", "style", "nav", "footer", "header"])
        {
            var nodes = doc.DocumentNode.SelectNodes($"//{tag}");
            if (nodes is not null)
                foreach (var node in nodes) node.Remove();
        }

        var content = doc.DocumentNode.SelectSingleNode("//main")
                   ?? doc.DocumentNode.SelectSingleNode("//article")
                   ?? doc.DocumentNode.SelectSingleNode("//body");

        if (content is null) return string.Empty;

        var sb = new StringBuilder();
        foreach (var node in content.DescendantsAndSelf())
        {
            if (node.NodeType != HtmlNodeType.Text) continue;
            string text = HtmlEntity.DeEntitize(node.InnerText).Trim();
            if (!string.IsNullOrWhiteSpace(text)) sb.AppendLine(text);
        }

        return sb.ToString();
    }

    public void Dispose() => EmbeddingModel.Dispose();
}
