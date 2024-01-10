from datasets import load_dataset, Dataset
import textstat

# Load the dataset
dataset = load_dataset("wikipedia", "20220301.simple")['train']

# Sort the dataset based on the length of the 'text' field in each example
sorted_dataset = sorted(dataset, key=lambda x: len(x['text']))

def get_flesch_score(example):
    return textstat.flesch_reading_ease(example['text'])

# Sort the dataset based on the Flesch Reading Ease score
# sorted_dataset = sorted(dataset, key=get_flesch_score, reverse=True)


def print_first_few_lines(text, num_lines=20):
    lines = text.split('\n')[:num_lines]
    for line in lines:
        print(line)
    print("...")

# Print the first 100 entries (articles)
for i, example in enumerate(sorted_dataset[:100]):
    article_text = example['text']
    print(f"Article {i + 1} Length: {len(article_text)} characters Title:{example['title']}")
    print_first_few_lines(article_text)  # Print the article content

    # Add a separator for better readability
    print("=" * 50)


# Convert the sorted list of examples back to a dataset object
sorted_dataset = Dataset.from_dict({key: [example[key] for example in sorted_dataset] for key in dataset.column_names})


# Save the sorted dataset to disk
sorted_dataset.save_to_disk("sorted_wikipedia_dataset")