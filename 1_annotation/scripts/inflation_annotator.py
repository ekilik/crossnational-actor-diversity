import pandas as pd
import csv
import re
import os
import argparse
from typing import List
from tqdm import tqdm
import ollama

class MainTopicAnnotator:
    def __init__(self, model_name: str = "phi4-mini:3.8b", seed: int = 0,
                 country_name: str = "the UK", country_key: str = "uk"):
        self.model_name = model_name
        self.seed = seed
        self.country_name = country_name
        self.country_key = country_key
        self.system_prompt = """
You are a professional news annotator specializing in identifying whether articles are mainly about increasing cost of living and if so, whether the discussion about this topic in the article is relevant to {country}. 
"""
        self.main_prompt = """
An article is considered mainly about increasing cost of living if the majority of its coverage focuses on this topic. Consider the complete article text. 

Include all forms of discussion related to the increasing cost of living, such as:
-	Inflation indicators: inflation rates, statistics and forecasts, changes in consumer price indices, retail or producer price indices, general consumer price developments, and impact of inflation on increasing cost of living,
-	Causes and contributing factors of rising prices/cost of living: government policies, global economic, political and financial developments, and changes in production or labor markets, inflation,
-	Discussion on interest rates (e.g. mortgages, loans, credit),
-	Impact of increasing prices/cost of living on the public in general, including different income and demographic groups and households,
-	Purchasing power and income: decreasing real wages/real pays due to inflation/increasing costs, income inequality due to rising prices, declining purchasing power, decreasing value of savings,
-	Costs and affordability of essential goods and services such as food, groceries, housing, energy (electricity, heating, gas), oil, fuel, transport, healthcare, childcare, and education,
-	Housing and mortgage costs: costs of rent, mortgage rates, house prices, borrowing costs, interest rates on mortgage loans, property market discussions, affordability of housing,
-	Energy prices: price changes (increasing or decreasing) in oil, fuel, electricity and gas,
-	Poverty and other affordability problems: People struggling to meet living costs, discussions about hunger and poverty, solutions for poverty, policies to reduce poverty or to help people with cost of living pressures,
-	Experiences and reactions from various groups within society (e.g. households, workers, union members, customers) due to rising prices, affordability, wages, cost of living pressures in their daily/working lives, including strikes, protests, personal stories,
-	Business and organizational impact: Effects of rising prices and inflation on business costs and operations, strategies to combat increasing costs/inflation, profitability, and workforce decisions, experiences of businesses, industry, and state organizations,
-	Labor relations and pay disputes: pay increase negotiations and wage demands, labor disputes and strikes,
-	Policy discussions/responses: interest rate discussions, central bank policies (incl. European Central Bank for EU countries) for inflation/interest rates, inflation targets, and government measures addressing rising prices or living costs.

Do **not** count brief mentions of increasing cost of living as main topic if the article is mainly about other issues (e.g. economic situation in the country (growth/recession/crisis), macroeconomic performance and policy (e.g. GDP, state debt), unless the coverage of such issues is directly linked and discussed in relation to increasing cost of living in the larger portion of the article text.

If the article is mainly about increasing cost of living, also decide whether the discussion is relevant to {country} context. When coding this variable, consider only the parts of the article about increasing cost of living and ignore the rest of the article. 

Code this variable as 1 if the article discusses increasing cost of living in {country} or in the world with implications for {country}. Code as 0 if the article discusses increasing cost of living in other countries without direct implications for {country}.

Use the following JSON format for your output. Your output should only contain the JSON object without any additional text or explanation.

```json
{{
  "article_id": "{article_id}",
  "main_topic_risingprices": 1 or 0,
  "relevant_to_{country_key}": 1 or 0
}}
```

Article to annotate:

Article ID: {article_id}
Text: {text}

Task: Using the annotation instructions, decide whether this article is mainly about increasing cost of living. Then, if the article is mainly about increasing cost of living, decide whether the discussion is relevant to {country} context.
- Provide a value (1 or 0) for "relevant_to_{country_key}" **only if main_topic_risingprices = 1**, otherwise set "relevant_to_{country_key}" to 0.
- Output your response in the provided JSON format, without any additional text or explanation.
"""

    def build_prompt(self, article_id: str, text: str) -> list[dict[str, str]]:
        """Create prompts in the format expected by Ollama"""
        formatted_main = self.main_prompt.format(article_id=article_id, text=text, country=self.country_name, country_key=self.country_key)
        messages = [
            {
                "role": "system",
                "content": self.system_prompt.format(country=self.country_name)
            },
            {
                "role": "user", 
                "content": formatted_main
            }
        ]
        return messages

    def regex_extract_to_dataframe(self, strings: List[str]) -> pd.DataFrame:
        """Extract structured data from LLM responses using regex"""
        article_ids = []
        about_risingprices_values = []
        relevant_to_country_values = []

        # Define regex pattern for article_id and about_risingprices with optional double quotes
        article_id_pattern = r'"article_id"\s*:\s*"?(\d+)"?'
        about_risingprices_pattern = rf'"main_topic_risingprices"\s*:\s*"?(\d)"?'
        relevant_to_country_pattern = rf'"relevant_to_{self.country_key}"\s*:\s*"?(\d)"?'

            # Iterate through each string
        for string_data in strings:
            # replace \ with nothing
            string_data = string_data.replace("\\", "")
            string_data = string_data.lower()
            
            # Use regex to find matches for article_id
            article_id_match = re.search(article_id_pattern, string_data)

            # Use regex to find matches for about_risingprices
            about_risingprices_match = re.search(about_risingprices_pattern, string_data)

            # Use regex to find matches for relevant_to_country
            relevant_to_country_match = re.search(relevant_to_country_pattern, string_data)

            # Extract values from the regex matches
            article_id = int(article_id_match.group(1)) if article_id_match else None
            about_risingprices = int(about_risingprices_match.group(1)) if about_risingprices_match else None
            relevant_to_country = int(relevant_to_country_match.group(1)) if relevant_to_country_match else None

            # Append values to the respective lists
            article_ids.append(article_id)
            about_risingprices_values.append(about_risingprices)
            relevant_to_country_values.append(relevant_to_country)

                # Create a DataFrame using the extracted values
        df = pd.DataFrame({
            "article_id": article_ids,
            "about_risingprices": about_risingprices_values,
            f"relevant_to_{self.country_key}": relevant_to_country_values
        })

        return df


    def chunk_dataframe(self, df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
        """Split dataframe into chunks for batch processing"""
        return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    def process_batch(self, batch_df: pd.DataFrame) -> List[str]:
        """Process a batch of articles and return raw responses"""
        results = []
        
        for idx, row in batch_df.iterrows():
            article_id = str(row['article_id'])
            text = row['text']

            messages = self.build_prompt(article_id, text)
            
            try:
                response = ollama.chat(
                    model=self.model_name,  
                    messages=messages,
                    options={
                        'temperature': 0.0, 
                        'seed': self.seed
                    }
                )
                
                response_content = response.message.content
                results.append(response_content)
                
            except Exception as e:
                print(f"Error processing article {article_id}: {str(e)}")
                # Create a fallback response
                fallback_response = f'{{"article_id": "{article_id}", "main_topic_risingprices": null, "relevant_to_{self.country_key}": null}}'
                results.append(fallback_response)
                
        return results

    def process_all_batches(self, df: pd.DataFrame, batch_size: int, 
                        output_file_path: str) -> None:
        """Process all batches and save results incrementally to one CSV file"""
        # Prepare unique articles
        unique_articles = df[['article_id', 'text']].drop_duplicates(subset=['article_id']).reset_index(drop=True)
        batches = self.chunk_dataframe(unique_articles, batch_size)
        
        # Remove existing output file if it exists to start fresh
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        
        # Track if we've written headers yet
        header_written = False
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            print(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} articles)")
            
            # Process only this batch
            batch_results = self.process_batch(batch)
            
            # Parse only this batch's results
            batch_coded_df = self.regex_extract_to_dataframe(batch_results)
            country_col = f"relevant_to_{self.country_key}"
            if country_col in batch_coded_df.columns:
                batch_coded_df.loc[batch_coded_df['about_risingprices'] == 0, country_col] = None
                                               
            # Append ONLY this batch to the file
            batch_coded_df.to_csv(
                output_file_path, 
                mode='a',  # Append mode - adds to end of existing file
                header=not header_written,  # Only write header on first batch
                index=False, 
                sep=';', 
                quoting=csv.QUOTE_NONNUMERIC
            )
            header_written = True
            
            print(f"Batch {batch_idx + 1} appended to {output_file_path}. Rising-prices articles in this batch: {batch_coded_df['about_risingprices'].sum()}")
        
        print(f"All batches completed! Check {output_file_path} for final results.")


def load_data(data_path: str, text_column: str = 'text',
             title_column: str = 'title', id_column: str = 'article_id') -> pd.DataFrame:
    """Load and prepare data from CSV"""
    df = pd.read_csv(data_path, encoding='utf-8-sig', sep=';', quoting=csv.QUOTE_NONNUMERIC)
    df.columns = [col.lower() for col in df.columns]
    df = df.rename(columns={id_column.lower(): 'article_id'})
    df['article_id'] = df['article_id'].astype(int)

    # make all colnames lowercase
    df.columns = [col.lower() for col in df.columns]
    
    df['text'] = df[title_column] + '\n' + df[text_column]
    
    return df

def main(args):
    # Pull model if needed
    print(f"Ensuring model {args.model_name} is available...")
    ollama.pull(args.model_name)
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    df = load_data(args.data_path, args.text_column, args.title_column, args.id_column)
    print(f"Loaded {len(df)} articles")

    # Initialize annotator
    annotator = MainTopicAnnotator(
        model_name=args.model_name,
        seed=args.seed,
        country_name=args.country_name,
        country_key=args.country_key
    )
    
    # Process all batches
    print(f"Starting batch processing with batch size {args.batch_size}...")
    annotator.process_all_batches(
        df=df,
        batch_size=args.batch_size,
        output_file_path=args.output_file_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process rising prices article annotations")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the CSV file containing articles')
    parser.add_argument('--output_file_path', type=str, required=True,
                       help='Path for the output CSV file')
    parser.add_argument('--id_column', type=str, default='article_id',
                       help='Name of the article ID column in the CSV')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of the text column in the CSV')
    parser.add_argument('--title_column', type=str, default='title',
                       help='Name of the title column in the CSV')
    parser.add_argument('--country_name', type=str, default='the UK',
                       help='Name of the country for relevance annotation')
    parser.add_argument('--country_key', type=str, default='uk',
                       help='Key for the country used in output column naming')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='phi4-mini:3.8b',
                       help='Name of the Ollama model to use')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Number of articles to process in each batch')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')

    args = parser.parse_args()
    main(args)