using System.Text;
using LMKit.Data;
using LMKit.Model;
using UglyToad.PdfPig;

namespace LmKitQuickStart.Rags;

/// <summary>
/// RAG pipeline for Samer's CV documents.
/// Indexes PDF files into a persisted DataSource with reranking enabled.
///
/// Usage:
///   using var rag = new RagSamerCv();
///   rag.EnsureIndexed();
///   var matches = rag.FindMatches(query);
///   var result  = rag.Answer(query, matches, chat);
/// </summary>
public sealed class RagSamerCv : RagEngineBase
{
    private static readonly string DocsFolder =
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Docs");

    private static readonly string IndexPath =
        Path.Combine(DocsFolder, "knowledge_base.dat");

    private static readonly string[] PdfFiles =
    [
        "SAMER ABD ALLAH ABD ALGANI.pdf",
        "Samer Abd Allah Abd Algani-MOBILR-WEB DEVELOPER.pdf",
        "SAMER ABD ALLAH ABD ALGANI-Technical Program Manager .pdf",
    ];

    private const string DataSourceName = "SamerCV";

    private readonly DataSource _dataSource;

    public RagSamerCv() : this(EmbeddingModelLoader.Load()) { }

    private RagSamerCv(LM embeddingModel)
        : base(embeddingModel, OpenOrCreate(IndexPath, DataSourceName, embeddingModel))
    {
        _dataSource = Engine.DataSources.First();
        Console.WriteLine(" Done.\n");
    }

    public void EnsureIndexed()
    {
        string docsFolder = Path.GetFullPath(DocsFolder);
        int indexed = 0, skipped = 0;

        foreach (string fileName in PdfFiles)
        {
            string section = Path.GetFileNameWithoutExtension(fileName);
            string fullPath = Path.Combine(docsFolder, fileName);

            if (_dataSource.HasSection(section))       { skipped++; continue; }
            if (!File.Exists(fullPath))                { Console.WriteLine($"  Not found: {fileName}"); continue; }

            Console.WriteLine($"  Indexing: {section}");
            Engine.ImportText(PdfToText(fullPath), DataSourceName, section);
            indexed++;
        }

        Console.WriteLine($"  Indexed {indexed} new, {skipped} cached. Total: {_dataSource.Sections.Count()} section(s).\n");
    }

    private static string PdfToText(string path)
    {
        var sb = new StringBuilder();
        using var doc = PdfDocument.Open(path);
        foreach (var page in doc.GetPages())
            sb.AppendLine(page.Text);
        return sb.ToString();
    }

    public override void Dispose() => EmbeddingModel.Dispose();
}
