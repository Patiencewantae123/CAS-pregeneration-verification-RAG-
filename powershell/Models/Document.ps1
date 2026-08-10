class Document {
    [string]$Id
    [string]$Content
    [string]$Source
    [double]$RetrievalConfidence
    [double]$RelevanceScore
    [double]$ProvenanceScore

    Document([string]$id, [string]$content, [string]$source, [double]$retrievalConf, [double]$relevance, [double]$provenance) {
        $this.Id                  = $id
        $this.Content             = $content
        $this.Source              = $source
        $this.RetrievalConfidence = $retrievalConf
        $this.RelevanceScore       = $relevance
        $this.ProvenanceScore      = $provenance
    }
}
