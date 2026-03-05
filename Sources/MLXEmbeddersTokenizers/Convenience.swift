// Copyright © Anthony DePasquale

import Foundation
import MLXEmbedders
import MLXLMCommon
import MLXLMTokenizers

/// Load an embedding model using ``TokenizersLoader`` for tokenization.
public func load(
    from downloader: any Downloader,
    configuration: MLXEmbedders.ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> (MLXEmbedders.EmbeddingModel, Tokenizer) {
    try await MLXEmbedders.load(
        from: downloader, using: TokenizersLoader(), configuration: configuration,
        useLatest: useLatest, progressHandler: progressHandler)
}

/// Load an embedding model container using ``TokenizersLoader`` for tokenization.
public func loadModelContainer(
    from downloader: any Downloader,
    configuration: MLXEmbedders.ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> MLXEmbedders.ModelContainer {
    try await MLXEmbedders.loadModelContainer(
        from: downloader, using: TokenizersLoader(), configuration: configuration,
        useLatest: useLatest, progressHandler: progressHandler)
}

/// Load an embedding model from a local directory using ``TokenizersLoader``
/// for tokenization.
public func load(
    from directory: URL
) async throws -> (MLXEmbedders.EmbeddingModel, Tokenizer) {
    try await MLXEmbedders.load(from: directory, using: TokenizersLoader())
}

/// Load an embedding model container from a local directory using
/// ``TokenizersLoader`` for tokenization.
public func loadModelContainer(
    from directory: URL
) async throws -> MLXEmbedders.ModelContainer {
    try await MLXEmbedders.loadModelContainer(from: directory, using: TokenizersLoader())
}
