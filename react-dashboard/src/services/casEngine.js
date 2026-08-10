export function runCASPipeline(documents, trustThreshold) {
  return documents.map((docA) => {
    let agreementScore = 0;
    let penaltyScore = 0;

    documents.forEach((docB) => {
      if (docA.id !== docB.id) {
        const textA = docA.content.toLowerCase();
        const textB = docB.content.toLowerCase();

        if (
          (textA.includes('approved') && textB.includes('rejected')) ||
          (textA.includes('rejected') && textB.includes('approved'))
        ) {
          penaltyScore += 0.95;
        } else if (textA.includes('approved') && textB.includes('approved')) {
          agreementScore += 0.88;
        }
      }
    });

    const trustScore = (0.35 * docA.rel) + (0.25 * docA.prov) + (0.20 * docA.conf) + (0.20 * Math.min(1.0, agreementScore));
    const finalWeight = trustScore - (0.50 * penaltyScore);
    const isFiltered = finalWeight < trustThreshold;

    return { ...docA, trustScore, penaltyScore, finalWeight, isFiltered };
  });
}
