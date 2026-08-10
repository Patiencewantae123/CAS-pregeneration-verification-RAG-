class TrustAwareSynthesizer {
    [double]$TrustThreshold

    TrustAwareSynthesizer([double]$threshold = 0.45) {
        $this.TrustThreshold = $threshold
    }

    [VerifiedDocument[]] FilterAndWeight([ConflictGraph]$graph) {
        $verifiedList = [System.Collections.Generic.List[VerifiedDocument]]::new()

        foreach ($nodeId in $graph.Nodes.Keys) {
            $doc = $graph.Nodes[$nodeId]
            $agreementFreq = $graph.GetAgreementScore($nodeId)
            $penalty       = $graph.GetContradictionPenalty($nodeId)

            $trustScore  = (0.35 * $doc.RelevanceScore) + (0.25 * $doc.ProvenanceScore) + (0.20 * $doc.RetrievalConfidence) + (0.20 * [Math]::Min(1.0, $agreementFreq))
            $finalWeight = $trustScore - (0.50 * $penalty)
            $isFiltered  = $finalWeight -lt $this.TrustThreshold

            $verifiedList.Add([VerifiedDocument]::new($doc, $trustScore, $penalty, $finalWeight, $isFiltered))
        }

        return $verifiedList.ToArray()
    }

    [string] SynthesizeContext([VerifiedDocument[]]$verifiedDocs) {
        $retainedDocs = $verifiedDocs | Where-Object { -not $_.IsFiltered } | Sort-Object -Property FinalWeight -Descending

        if ($retainedDocs.Count -eq 0) {
            return "[CRITICAL WARNING]: All evidence nodes were rejected due to severe contradiction density."
        }

        $sb = [System.Text.StringBuilder]::new()
        [void]$sb.AppendLine("=== VERIFIED RECONCILED CONTEXT (C_verified) ===")
        foreach ($vDoc in $retainedDocs) {
            [void]$sb.AppendLine(("[Source: {0} | Net Weight: {1:N2}] {2}" -f $vDoc.Doc.Source, $vDoc.FinalWeight, $vDoc.Doc.Content))
        }
        return $sb.ToString()
    }
}
