import argparse
from ingestion import load_data, merge_data
from transform import impute_missing, transform_numeric, encode_categoricals
from deduplication import clean_dataframe, fuzzy_deduplicate, semantic_deduplicate

def main(args):
    # Load main data
    df = load_data(args.input)

    # Merge if required
    if args.merge and args.on:
        merge_columns = [col.strip() for col in args.on.split(',')]
        df2 = load_data(args.merge)
        df = merge_data(df, df2, merge_columns)

    # Clean text & remove emojis/junk characters
    if args.clean:
        df = clean_dataframe(df)

    # Drop exact duplicates
    if args.deduplicate:
        df.drop_duplicates(inplace=True)

    # Impute missing values
    if args.impute:
        df = impute_missing(df, strategy=args.impute)

    # Transform numeric columns
    if args.transform:
        df = transform_numeric(df, method=args.transform)

    # Encode categorical features
    if args.encode:
        df = encode_categoricals(df, method=args.encode)

    # Fuzzy text deduplication
    if args.fuzzy_dedup:
        df = fuzzy_deduplicate(df, threshold=args.fuzzy_threshold)

    # Semantic deduplication
    if args.semantic_dedup:
        df = semantic_deduplicate(df, column=args.semantic_column, threshold=args.semantic_threshold)

    # Save output
    df.to_csv(args.output, index=False)
    print(f"✅ Preprocessing complete. Output saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🔧 Modular Preprocessing Pipeline")

    parser.add_argument('--input', required=True, help='Input data file (.csv/.xlsx/.json)')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--merge', help='Second dataset to merge')
    parser.add_argument('--on', help='Comma-separated column names to merge on')

    parser.add_argument('--clean', action='store_true', help='Clean text: junk chars, punctuation, emojis')
    parser.add_argument('--deduplicate', action='store_true', help='Drop exact duplicate rows')

    parser.add_argument('--impute', choices=['mean', 'median', 'mode', 'knn'], help='Missing value strategy')
    parser.add_argument('--transform', choices=['normalize', 'standardize', 'log'], help='Numeric transformation')
    parser.add_argument('--encode', choices=['onehot', 'label'], help='Categorical encoding method')

    parser.add_argument('--fuzzy_dedup', action='store_true', help='Enable fuzzy deduplication')
    parser.add_argument('--fuzzy_threshold', type=int, default=90, help='Fuzzy match threshold (0–100)')

    parser.add_argument('--semantic_dedup', action='store_true', help='Enable semantic deduplication')
    parser.add_argument('--semantic_column', default='sentence', help='Column to apply semantic deduplication on')
    parser.add_argument('--semantic_threshold', type=float, default=0.9, help='Cosine similarity threshold (0–1)')

    args = parser.parse_args()
    main(args)
