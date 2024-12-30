import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
_open_ai_tkn = os.environ.get('OPENAI_KEY')
_project_tkn = os.environ.get('OPENAI_PROJECT')
_organisation_tkn = os.environ.get('OPENAI_ORG')

client = OpenAI(organization=_organisation_tkn, project=_project_tkn, api_key=_open_ai_tkn)

# Constants
DATASET_PATH = "dataset.parquet"
TEMPLATES_PATH = "templates.txt"
OUTPUT_FILE = "output.xlsx" 
MODEL = "gpt-4o-mini"
CHUNK_SIZE = 24  # This is the number of songs per template

# Load dataset and templates
dataset = pd.read_parquet(DATASET_PATH).sample(120, random_state=383)  # 5 * 24 rows
with open(TEMPLATES_PATH) as f:
    templates = f.readlines()

# Function to process a chunk of songs with a specific template
def process_chunk(chunk, template):
    results = []
    for _, row in chunk.iterrows():
        prompt = template.replace("[LYR]", row['lyrics']).strip()
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL
            )
            results.append({
                "artist": row['artist'],
                "title": row['title'],
                "response": response.choices[0].message.content,
                "template_used": template
            })
        except Exception as e:
            print(f"Error processing {row['title']}: {str(e)}")
    return results

# Process song lyrics in chunks, 24 (or CHUNK_SIZE) songs per template
results = []
for i, template in enumerate(templates):
    start_idx = i * CHUNK_SIZE
    end_idx = start_idx + CHUNK_SIZE
    chunk = dataset.iloc[start_idx:end_idx]
    print(f"Processing rows {start_idx} to {end_idx} with template {i + 1}")
    results.extend(process_chunk(chunk, template))

# Save to excel file
pd.DataFrame(results).to_excel(OUTPUT_FILE, index=False)
print(f"Processing complete. Results saved to {OUTPUT_FILE}")