enum NLIRelation {
    Entailment
    Contradiction
    Neutral
}

class NLIEdge {
    [string]$SourceDocId
    [string]$TargetDocId
    [NLIRelation]$Relation
    [double]$Confidence

    NLIEdge([string]$src, [string]$target, [NLIRelation]$rel, [double]$conf) {
        $this.SourceDocId = $src
        $this.TargetDocId = $target
        $this.Relation    = $rel
        $this.Confidence  = $conf
    }
}
