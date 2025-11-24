from typing import List, Optional, Tuple
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

LABEL_DIST_PNG = 'label_distribution.png'
LENGTH_DIST_PNG = 'length_distribution.png'
MODEL_COMP_CSV = 'model_comparison.csv'
MODEL_F1_PNG = 'model_f1_scores.png'
CLEANED_CSV = 'spam_clean.csv'

MODEL_FILE = 'best_model.joblib'
TFIDF_FILE = 'tfidf_vectorizer.joblib'
LABEL_ENCODER_FILE = 'label_encoder.joblib'
MODEL_META = 'model_meta.joblib'

def load_and_clean(path: str) -> Tuple[pd.DataFrame, LabelEncoder]:
	df = pd.read_csv(path, encoding='latin-1')

	for col in ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']:
		if col in df.columns:
			df = df.drop(columns=[col])

	df = df.rename(columns={'v1': 'target', 'v2': 'text'})

	le = LabelEncoder()
	df['target'] = le.fit_transform(df['target'])

	df = df.dropna().drop_duplicates(keep='first')

	df.to_csv(CLEANED_CSV, index=False)

	return df, le

def visualize(df: pd.DataFrame) -> None:
	sns.set(style='whitegrid')

	plt.figure(figsize=(6, 4))
	ax = sns.countplot(x=df['target'])
	ax.set_xticks([0, 1])
	ax.set_xticklabels(['ham', 'spam'])
	plt.title('Label distribution')
	plt.tight_layout()
	plt.savefig(LABEL_DIST_PNG)
	plt.close()

	df['length'] = df['text'].str.len()
	plt.figure(figsize=(8, 4))
	sns.histplot(data=df, x='length', hue=df['target'], bins=50, element='step', stat='density', common_norm=False)
	plt.title('Message length distribution by label')
	plt.tight_layout()
	plt.savefig(LENGTH_DIST_PNG)
	plt.close()

def prepare_features(df: pd.DataFrame) -> Tuple[TfidfVectorizer, any, any]:
	tfidf = TfidfVectorizer(stop_words='english', max_df=0.9)
	X = tfidf.fit_transform(df['text'])
	y = df['target']
	return tfidf, X, y

def train_and_evaluate(X, y) -> Tuple[dict, pd.DataFrame]:
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

	models = {
		'MultinomialNB': MultinomialNB(),
		'LogisticRegression': LogisticRegression(max_iter=1000),
		'LinearSVC': LinearSVC(max_iter=10000),
		'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
	}

	records = []
	for name, model in models.items():
		model.fit(X_train, y_train)
		preds = model.predict(X_test)
		acc = accuracy_score(y_test, preds)
		prec = precision_score(y_test, preds, zero_division=0)
		rec = recall_score(y_test, preds, zero_division=0)
		f1 = f1_score(y_test, preds, zero_division=0)
		records.append({'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})

	results = pd.DataFrame(records).sort_values(by='f1', ascending=False)
	results.to_csv(MODEL_COMP_CSV, index=False)

	plt.figure(figsize=(7, 4))
	sns.barplot(x='model', y='f1', data=results)
	plt.ylim(0, 1)
	plt.tight_layout()
	plt.savefig(MODEL_F1_PNG)
	plt.close()

	return models, results

def choose_best_model(models: dict, results: pd.DataFrame):
	best_name = results.iloc[0]['model']
	return models[best_name], best_name

def detect_messages(messages: List[str], model, vectorizer, label_encoder: LabelEncoder) -> List[dict]:
	X_new = vectorizer.transform(messages)
	preds = model.predict(X_new)
	probs = None
	if hasattr(model, 'predict_proba'):
		probs = model.predict_proba(X_new)[:, 1]

	decoded = label_encoder.inverse_transform(preds)
	out = []
	for i, text in enumerate(messages):
		out.append({'text': text, 'predicted_label': decoded[i], 'probability_spam': float(probs[i]) if probs is not None else None})
	return out

def main(dataset_path: str, input_text: Optional[str], verbose: bool, save_model: bool=False, load_model: bool=False):
	df, label_encoder = load_and_clean(dataset_path)

	visualize(df)

	tfidf, X, y = prepare_features(df)

	best_model = None
	best_name = None
	results = None

	if load_model and os.path.exists(MODEL_FILE) and os.path.exists(TFIDF_FILE) and os.path.exists(LABEL_ENCODER_FILE):
		best_model = joblib.load(MODEL_FILE)
		tfidf = joblib.load(TFIDF_FILE)
		label_encoder = joblib.load(LABEL_ENCODER_FILE)
		if os.path.exists(MODEL_META):
			try:
				meta = joblib.load(MODEL_META)
				best_name = meta.get('model_name', 'loaded_model')
				results = meta.get('results_df')
			except Exception:
				best_name = 'loaded_model'
		else:
			best_name = 'loaded_model'
	else:
		models, results = train_and_evaluate(X, y)
		best_model, best_name = choose_best_model(models, results)

		if save_model:
			joblib.dump(best_model, MODEL_FILE)
			joblib.dump(tfidf, TFIDF_FILE)
			joblib.dump(label_encoder, LABEL_ENCODER_FILE)
			meta = {'model_name': best_name, 'results_df': results}
			joblib.dump(meta, MODEL_META)

	if verbose:
		model_f1 = results.iloc[0]['f1'] if results is not None else None
		counts = df['target'].value_counts().to_dict()
		ham_count = counts.get(0, 0)
		spam_count = counts.get(1, 0)
		length_stats = df.groupby('target')['length'].agg(['mean', 'median']).to_dict()
		if model_f1 is not None:
			print(f"Model: {best_name} (F1={model_f1:.3f})")
		else:
			print(f"Model: {best_name} (loaded)")
		print(f"Labels: ham={ham_count}, spam={spam_count}")
		print(f"Message length (mean,median) by label:")
		print(f"  ham: {length_stats['mean'].get(0,'-'):.1f}, {length_stats['median'].get(0,'-'):.1f}")
		print(f"  spam: {length_stats['mean'].get(1,'-'):.1f}, {length_stats['median'].get(1,'-'):.1f}")
		files = [CLEANED_CSV, LABEL_DIST_PNG, LENGTH_DIST_PNG, MODEL_COMP_CSV, MODEL_F1_PNG]
		if save_model or load_model:
			files += [MODEL_FILE, TFIDF_FILE, LABEL_ENCODER_FILE, MODEL_META]
		print(f"Saved files: {', '.join(files)}")

	if input_text:
		res = detect_messages([input_text], best_model, tfidf, label_encoder)
		r = res[0]
		model_name = best_name
		model_f1 = results.iloc[0]['f1'] if results is not None else None
		label = r['predicted_label']
		prob = r['probability_spam']
		if prob is not None:
			if model_f1 is not None:
				print(f"Prediction: {label}\nProbability(spam): {prob:.2f}\nModel: {model_name} (F1={model_f1:.3f})")
			else:
				print(f"Prediction: {label}\nProbability(spam): {prob:.2f}\nModel: {model_name} (loaded)")
		else:
			if model_f1 is not None:
				print(f"Prediction: {label}\nModel: {model_name} (F1={model_f1:.3f})")
			else:
				print(f"Prediction: {label}\nModel: {model_name} (loaded)")
		return

	try:
		line = input().strip()
		if line == '':
			return
		res = detect_messages([line], best_model, tfidf, label_encoder)
		r = res[0]
		model_name = best_name
		model_f1 = results.iloc[0]['f1'] if results is not None else None
		label = r['predicted_label']
		prob = r['probability_spam']
		if prob is not None:
			if model_f1 is not None:
				print(f"Prediction: {label} | Prob(spam): {prob:.2f} | Model: {model_name} (F1={model_f1:.3f})")
			else:
				print(f"Prediction: {label} | Prob(spam): {prob:.2f} | Model: {model_name} (loaded)")
		else:
			if model_f1 is not None:
				print(f"Prediction: {label} | Model: {model_name} (F1={model_f1:.3f})")
			else:
				print(f"Prediction: {label} | Model: {model_name} (loaded)")
	except (EOFError, KeyboardInterrupt):
		return

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Simple SMS spam detector (minimal output)')
	parser.add_argument('--text', '-t', type=str, help='Provide a single SMS text to classify')
	parser.add_argument('--verbose', '-v', action='store_true', help='Show brief progress and saved artifact names')
	parser.add_argument('--dataset', '-d', type=str, default='spam.csv', help='Path to spam.csv dataset')
	parser.add_argument('--save-model', action='store_true', help='Save the trained best model, vectorizer and label encoder')
	parser.add_argument('--load-model', action='store_true', help='Load a previously saved model and skip training')
	args = parser.parse_args()

	main(dataset_path=args.dataset, input_text=args.text, verbose=args.verbose, save_model=args.save_model, load_model=args.load_model)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Simple SMS spam detector (minimal output)')
	parser.add_argument('--text', '-t', type=str, help='Provide a single SMS text to classify')
	parser.add_argument('--verbose', '-v', action='store_true', help='Show brief progress and saved artifact names')
	parser.add_argument('--dataset', '-d', type=str, default='spam.csv', help='Path to spam.csv dataset')
	parser.add_argument('--save-model', action='store_true', help='Save the trained best model, vectorizer and label encoder')
	parser.add_argument('--load-model', action='store_true', help='Load a previously saved model and skip training')
	args = parser.parse_args()

	main(dataset_path=args.dataset, input_text=args.text, verbose=args.verbose, save_model=args.save_model, load_model=args.load_model)