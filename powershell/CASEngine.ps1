. "$PSScriptRoot/Models/Document.ps1"
. "$PSScriptRoot/Models/NLIEdge.ps1"
. "$PSScriptRoot/Models/VerifiedDocument.ps1"
. "$PSScriptRoot/Services/NLIVerifier.ps1"
. "$PSScriptRoot/Services/ConflictGraph.ps1"
. "$PSScriptRoot/Services/TrustAwareSynthesizer.ps1"

class CASEngine {
    [NLIVerifier]$NLI
    [TrustAwareSynthesizer]$Synthesizer

    CASEngine([double]$trustThreshold = 0.45) {
        $this.NLI         = [NLIVerifier]::new()
        $this.Synthesizer = [TrustAwareSynthesizer]::new($trustThreshold)
    }

    [hashtable] ExecutePipeline([string]$query, [Document[]]$retrievedDocs) {
        $edges = $this.NLI.BuildPairwiseMatrix($retrievedDocs)
        $graph = [ConflictGraph]::new($retrievedDocs, $edges)
        $verifiedDocs = $this.Synthesizer.FilterAndWeight($graph)
        $cVerified = $this.Synthesizer.SynthesizeContext($verifiedDocs)

        return @{
            Query             = $query
            Graph             = $graph
            VerifiedDocuments = $verifiedDocs
            C_Verified        = $cVerified
        }
    }

    static [double] ComputeCPI([double]$acc, [double]$cons, [double]$hall, [double]$contra) {
        return ($acc + $cons + (100.0 - $hall) + (100.0 - $contra)) / 4.0
    }
}
