using LMKit.Data;
using LMKit.Model;
using LMKit.Retrieval;
using LMKit.TextGeneration;
using LMKit.TextGeneration.Chat;

namespace LmKitQuickStart.Rags;

/// <summary>
/// Base class for all RAG pipelines.
/// Handles RagEngine setup, reranking, DataSource resilience, and answer generation.
/// Subclasses provide the DataSource and define how documents are indexed.
/// </summary>
public abstract class RagEngineBase : IDisposable
{
    protected const int    DefaultTopK     = 5;
    protected const float  DefaultMinScore  = 0.2f;
    protected const float  RerankAlpha      = 0.7f;
    public    const int    ChunkSize        = 500;
    public    const int    ChunkOverlap     = 50;

    protected readonly LM        EmbeddingModel;
    protected readonly RagEngine Engine;

    protected RagEngineBase(LM embeddingModel, DataSource dataSource)
    {
        EmbeddingModel = embeddingModel;

        Engine = new RagEngine(embeddingModel);
        Engine.AddDataSource(dataSource);
        Engine.DefaultIChunking = new TextChunking { MaxChunkSize = ChunkSize, MaxOverlapSize = ChunkOverlap };
        Engine.Reranker         = new RagEngine.RagReranker(embeddingModel, rerankedAlpha: RerankAlpha);
    }

    public IList<PartitionSimilarity> FindMatches(string query, int topK = DefaultTopK, float minScore = DefaultMinScore) =>
        Engine.FindMatchingPartitions(query, topK, minScore);

    public TextGenerationResult Answer(string query, IList<PartitionSimilarity> matches, SingleTurnConversation chat) =>
        Engine.QueryPartitions(query, matches, chat);

    protected static DataSource OpenOrCreate(string path, string name, LM model)
    {
        if (File.Exists(path))
        {
            try   { return DataSource.LoadFromFile(path, readOnly: false); }
            catch { Console.WriteLine("  [Index corrupt — rebuilding...]"); File.Delete(path); }
        }
        return DataSource.CreateFileDataSource(path, name, model);
    }

    public virtual void Dispose() { }
}
