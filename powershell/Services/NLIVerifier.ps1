class NLIVerifier {
    [NLIEdge] CompareDocuments([Document]$docA, [Document]$docB) {
        $textA = $docA.Content.ToLower()
        $textB = $docB.Content.ToLower()

        $hasA_Approve = ($textA -match "approved" -or $textA -match "increased" -or $textA -match "effective")
        $hasB_Reject  = ($textB -match "rejected" -or $textB -match "decreased" -or $textB -match "ineffective")
        $hasB_Approve = ($textB -match "approved" -or $textB -match "increased" -or $textB -match "effective")
        $hasA_Reject  = ($textA -match "rejected" -or $textA -match "decreased" -or $textA -match "ineffective")

        if (($hasA_Approve -and $hasB_Reject) -or ($hasA_Reject -and $hasB_Approve)) {
            return [NLIEdge]::new($docA.Id, $docB.Id, [NLIRelation]::Contradiction, 0.95)
        }

        $wordsA = $textA.Split(' ') | Where-Object { $_.Length -gt 3 }
        $wordsB = $textB.Split(' ') | Where-Object { $_.Length -gt 3 }

        if ($wordsA.Count -gt 0) {
            $intersection = $wordsA | Where-Object { $wordsB -contains $_ }
            if (($intersection.Count / $wordsA.Count) -gt 0.35) {
                return [NLIEdge]::new($docA.Id, $docB.Id, [NLIRelation]::Entailment, 0.88)
            }
        }

        return [NLIEdge]::new($docA.Id, $docB.Id, [NLIRelation]::Neutral, 0.50)
    }

    [System.Collections.Generic.List[NLIEdge]] BuildPairwiseMatrix([Document[]]$documents) {
        $edges = [System.Collections.Generic.List[NLIEdge]]::new()
        for ($i = 0; $i -lt $documents.Length; $i++) {
            for ($j = $i + 1; $j -lt $documents.Length; $j++) {
                $edges.Add($this.CompareDocuments($documents[$i], $documents[$j]))
            }
        }
        return $edges
    }
}
