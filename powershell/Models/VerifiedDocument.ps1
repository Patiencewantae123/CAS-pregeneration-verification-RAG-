class VerifiedDocument {
    [Document]$Doc
    [double]$TrustScore
    [double]$ConflictPenalty
    [double]$FinalWeight
    [bool]$IsFiltered

    VerifiedDocument([Document]$doc, [double]$trust, [double]$penalty, [double]$weight, [bool]$filtered) {
        $this.Doc             = $doc
        $this.TrustScore      = $trust
        $this.ConflictPenalty = $penalty
        $this.FinalWeight     = $weight
        $this.IsFiltered      = $filtered
    }
}
