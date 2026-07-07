using LMKit.Model;
using LMKit.Retrieval;
using LMKit.TextGeneration.Chat;
using LMKit.TextGeneration.Sampling;

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
        "If the context does not contain the answer, say so. " +
        "Always reply in the same language the user wrote in.";

    public RagAbpDocs(AbpDocsKnowledgeBase kb) : base(kb.EmbeddingModel, kb.DataSource) { }

    public RagChat CreateRagChat(LM chatModel) => new(Engine, chatModel)
    {
        QueryGenerationMode     = QueryGenerationMode.MultiQuery,
        MaxRetrievedPartitions  = DefaultTopK,
        MinRelevanceScore       = DefaultMinScore,
        SystemPrompt            = SystemPrompt,
        MaximumCompletionTokens = 512,
        SamplingMode            = new RandomSampling { Temperature = 0.7f, TopP = 0.9f, TopK = 40 },
        RepetitionPenalty       = { TokenCount = 256, RepeatPenalty = 1.3f, PresencePenalty = 0.6f },
    };
}
