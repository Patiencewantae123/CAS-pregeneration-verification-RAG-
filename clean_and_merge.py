import glob
import json
import os

def normalize_sample(dataset_name, sample, idx):
    # Standard schema definition
    dataset_tag = dataset_name.replace('_hf_samples.json', '').replace('.json', '')
    sample_id = sample.get('id') or sample.get('_id') or f'{dataset_tag}_{idx:04d}'
    
    # 1. Normalize Query / Question / Claim
    query = (
        sample.get('query') or 
        sample.get('question') or 
        sample.get('ambiguous_question') or 
        sample.get('instruction') or 
        sample.get('claim') or 
        sample.get('prompt') or ''
    )
    
    # 2. Normalize Contexts / Evidence / Retrieved Documents
    raw_contexts = (
        sample.get('retrieved_contexts') or 
        sample.get('retrieved_passages') or 
        sample.get('retrieved_documents') or 
        sample.get('context') or 
        sample.get('evidence') or 
        sample.get('domain_documents') or 
        sample.get('knowledge') or []
    )
    
    contexts = []
    if isinstance(raw_contexts, list):
        for item in raw_contexts:
            if isinstance(item, str):
                contexts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get('text') or item.get('snippet') or str(item)
                contexts.append(text.strip())
            else:
                contexts.append(str(item))
    elif isinstance(raw_contexts, str):
        contexts = [raw_contexts.strip()]
        
    # 3. Normalize Ground Truth / Answer / Label
    ground_truth = (
        sample.get('ground_truth') or 
        sample.get('synthesized_response') or 
        sample.get('gold_answer') or 
        sample.get('right_answer') or 
        sample.get('disambiguated_answers') or 
        sample.get('answer') or 
        sample.get('label') or ''
    )
    
    return {
        'id': str(sample_id),
        'dataset': dataset_tag,
        'query': str(query).strip(),
        'contexts': contexts,
        'ground_truth': str(ground_truth).strip(),
        'metadata': sample.get('metadata', {})
    }

def load_records(file_path):
    # The *_hf_samples.json files are JSON Lines, the others are JSON arrays
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

def process_and_merge():
    # Find all sample JSON files in the data folder
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    patterns = ['*_samples.json', '*qa.json', '*rag*.json']
    json_files = sorted({f for p in patterns for f in glob.glob(os.path.join(data_dir, p))})
    
    merged_data = []
    seen_ids = set()
    total_cleaned = 0
    
    print(f'Found {len(json_files)} dataset JSON files to process.')
    
    for file_path in json_files:
        try:
            content = load_records(file_path)
            samples = content if isinstance(content, list) else [content]
            file_count = 0
            
            for idx, raw_sample in enumerate(samples, start=1):
                clean_record = normalize_sample(os.path.basename(file_path), raw_sample, idx)
                
                # Deduplication check: ids alone are not unique (ConflictBank ids are
                # Wikidata relation ids shared by many claims), so key on id + query
                key = (clean_record['id'], clean_record['query'])
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                
                merged_data.append(clean_record)
                file_count += 1
                
            print(f'  [?] {file_path}: Processed {file_count} records')
            total_cleaned += file_count
        except Exception as e:
            print(f'  [X] Could not process {file_path}: {e}')

    # Output unified dataset
    output_filename = os.path.join(os.path.dirname(data_dir), 'unified_benchmark_dataset.json')
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2)
        
    print(f'\nSuccess! Merged {total_cleaned} clean samples across datasets into \"{output_filename}\".')

if __name__ == '__main__':
    process_and_merge()
