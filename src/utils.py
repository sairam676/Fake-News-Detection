import pandas as pd

def load_and_clean(fake_path, true_path):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    df = df[['title', 'text', 'label']].dropna()

    df['content'] = df['title'] + " " + df['text']

    return df[['content', 'label']]
