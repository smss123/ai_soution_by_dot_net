using LMKit.Model;
using LMKit.Retrieval;
using LMKit.TextGeneration.Chat;

namespace LmKitQuickStart.Rags;

/// <summary>
/// RAG pipeline for ABP Framework documentation.
/// Attaches to a shared AbpDocsKnowledgeBase with reranking enabled.
///
/// Usage:
///   using var kb  = new AbpDocsKnowledgeBase();
///   await kb.EnsureIndexedAsync();
///   var rag       = new RagAbpDocs(kb);
///   using var chat = rag.CreateRagChat(chatModel);
///   var result    = chat.Submit(query);
/// </summary>
public sealed class RagAbpDocs : RagEngineBase
{
    private const string SystemPrompt =
        "You are an ABP Framework expert. Answer using only the provided context. " +
        "If the context does not contain the answer, say so.";

    public RagAbpDocs(AbpDocsKnowledgeBase kb) : base(kb.EmbeddingModel, kb.DataSource) { }

    public RagChat CreateRagChat(LM chatModel) => new(Engine, chatModel)
    {
        QueryGenerationMode     = QueryGenerationMode.MultiQuery,
        MaxRetrievedPartitions  = DefaultTopK,
        MinRelevanceScore       = DefaultMinScore,
        SystemPrompt            = SystemPrompt,
        MaximumCompletionTokens = 512,
    };
}
