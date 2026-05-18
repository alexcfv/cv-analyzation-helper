class IndexService:
    def __init__(self, embedder, loader, vector_store, profile_builder, profile_repository):
        self.embedder = embedder
        self.loader = loader
        self.vector_store = vector_store
        self.profile_builder = profile_builder
        self.profile_repository = profile_repository

    def index_folder(self, dir_path: str) -> dict:
        documents = self.loader.load_folder(dir_path)

        existing = set()
        if self.vector_store.collection.count() > 0:
            existing = set(
                m["source"]
                for m in self.vector_store.collection.get(limit=10000)["metadatas"]
            )

        new_docs = [d for d in documents if d["source"] not in existing]

        if new_docs:
            self.vector_store.add_documents(new_docs, self.embedder)

        all_grouped = self.vector_store.get_all_grouped_by_source()
        new_profiles = []
        for source, chunks in all_grouped.items():
            if not self.profile_repository.profile_exists(source):
                profile = self.profile_builder.build_profile(chunks)
                self.profile_repository.create_profile(source, profile)
                new_profiles.append(source)

        return {
            "total_files": len(set(d["source"] for d in documents)),
            "new_chunks": len(new_docs),
            "new_profiles": new_profiles,
        }
