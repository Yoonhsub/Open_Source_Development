import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# �ſ�ī�带 ���� ���� �ŷ� ���� �̿����� ����鿡 ���� �����ͼ� ����
data = {
    'Customer_ID' : [4524, 2731, 3701, 42, 4659, 4537, 1278, 3687, 3465, 3754, 4820, 5731, 6145, 7123, 8456, 9234, 1025, 1138, 1247, 1358],
    'Name' : ['Lisa', 'Madison', 'John', 'Chris', 'Tevez', 'Lusso', 'Mira', 'Chen', 'Maira', 'Kim', 'Alex', 'Bella', 'Cathy', 'David', 'Eva', 'Frank', 'Grace', 'Hank', 'Ivy', 'Jack'],
    'Age' : [29, 34, 23, 40, 31, 44, 24, 25, 28, 51, 30, 22, 35, 29, 33, 41, 26, 38, 27, 32],
    'Email' : ['Rosy0129@gmail.com', 'Mads8891@gmail.com', 'urbancoo12@gmail.com', 'Chizz667@gmail.com', 'Tobi1801@gmail.com', 
               'Lussyoi77@gmail.com', 'Blackmi78@gmail.com', 'Korchen1@gmail.com', 'Lunasol8920@gmail.com', 'JKim1235@gmail.com',
               'alex@gmail.com', 'bella@gmail.com', 'cathy@gmail.com', 'david@gmail.com', 'eva@gmail.com', 'frank@gmail.com', 
               'grace@gmail.com', 'hank@gmail.com', 'ivy@gmail.com', 'jack@gmail.com'],
    'Card_brand' : ['Visa', 'Visa', 'Visa', 'Visa', 'Mastercard', 'Mastercard', 'Discover','Visa', 'Amex', 'Mastercard', 
                    'Visa', 'Mastercard', 'Discover', 'Amex', 'Visa', 'Mastercard', 'Discover', 'Visa', 'Amex', 'Mastercard'],
    'Card_type' : ['Credit', 'Debit', 'Credit', 'Credit', 'Debit', 'Credit', 'Debit', 'Debit', 'Credit', 'Credit', 
                   'Credit', 'Debit', 'Credit', 'Credit', 'Debit', 'Credit', 'Debit', 'Credit', 'Debit', 'Credit'],
    'Expire_date' : ['2025-12', '2026-11', '2026-03', '2027-04', '2025-05', '2025-11', '2026-01', '2027-08', '2026-09', '2028-07',
                     '2025-08', '2026-09', '2026-11', '2028-01', '2025-12', '2025-10', '2026-05', '2027-07', '2026-03', '2028-04']
}

df = pd.DataFrame(data)
df['Expire_date'] = pd.to_datetime(df['Expire_date'] + '-01', format='%Y-%m-%d')

# ī��縦 �������� ������ �׷�ȭ
grouped = df.groupby('Card_brand')

# �׷�ȭ�� ���� ���ȭ�� ������
count_customers = grouped['Customer_ID'].count()  # �� ��
mean_age = grouped['Age'].mean()  # ��� ����
unique_card_types = grouped['Card_type'].nunique()  # ī�� ���� ��

# ��� ����
summary = pd.DataFrame({
    'Total_Customer': count_customers,
    'Average_Age': mean_age,
    'Card_Types': unique_card_types
})

print(summary)

# �׷��� 1: ī��纰 �� ��
plt.figure(figsize=(10, 5))
sns.barplot(x=summary.index, y=summary['Total_Customer'])
plt.title('ī��纰 �� ��')
plt.xlabel('Card_brand')
plt.ylabel('Total_Customer')
plt.xticks(rotation=45)
plt.show()

# �׷��� 2: ī��纰 �� ��� ����
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], bins=10, kde=True)  # ������׷� ����
plt.title('�� ���� ����')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.axvline(df['Age'].mean(), color='r', linestyle='dashed', linewidth=1)  # ��� ���� �� �߰�
plt.text(df['Age'].mean() + 1, 2, f'Mean: {df["Age"].mean():.2f}', color='r')  # ��� ���� �ؽ�Ʈ �߰�
plt.show()

# �ӽŷ��� ��
# ī���� ī�������� ���ڷ� ���ڵ�
le_card_company = LabelEncoder()
le_card_type = LabelEncoder()
df['Card_brand'] = le_card_company.fit_transform(df['Card_brand'])
df['Card_type'] = le_card_type.fit_transform(df['Card_type'])

# Ư���� Ÿ�� ���� ����
X = df[['Card_brand', 'Card_type']]  # ���� ����
y = df['Age']  # ���� ����

# �Ʒ� ��Ʈ�� �׽�Ʈ ��Ʈ�� ������ ����
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# �� �н�
model = LinearRegression()
model.fit(X_train, y_train)

# ����
y_pred = model.predict(X_test)

# �� ��
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
