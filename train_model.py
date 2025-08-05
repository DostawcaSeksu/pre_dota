import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('dota_preprocessed_data.csv', index_col='match_id')
    print('processed data loaded successfully')
except FileNotFoundError:
    print('Error: File "dota_preprocessed_data.csv" was not found.')
    exit()

x = df.drop('radiant_win', axis=1)
y = df['radiant_win']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print('\n Scaling data...')

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print('\nData was scaled successfully.')

print(f'\nTrain sample size: {x_train_scaled.shape[0]} matches')
print(f'\nTest sample size: {x_test_scaled.shape[0]} matches')

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print('\nStarting model training...')
model.fit(x_train_scaled, y_train)
print('Model was trained successfully')

y_pred = model.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f'\nModel accurasy on test data: {accuracy:.2%}')
print('\n Detail classification report: ')
print(classification_report(y_test, y_pred, target_names=['Dire_win', 'Radiant Win']))

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': x.columns,
    'importance': importances
}).sort_values('importance', ascending=False)
print('\nValues importance for win prediction: ')
print(feature_importance_df)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dire Win', 'Radiant Win'], yticklabels=['Dire Win', 'Radiant Win'])
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Error matrix for random forest')
plt.show()