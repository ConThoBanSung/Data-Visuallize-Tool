# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, FalconForCausalLM, AutoTokenizer

# Function to visualize the dataset
def visualize_data(df):
    print("Đang vẽ biểu đồ...")
    df.hist(bins=30, figsize=(10, 8))
    plt.tight_layout()
    plt.show()

    if len(df.columns) > 1:
        sns.pairplot(df)
        plt.show()

    if df.select_dtypes(include='number').shape[1] > 1:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.title("Heatmap of Correlation")
        plt.show()

# Preliminary data analysis
def analyze_data(df):
    print("Phân tích sơ bộ dữ liệu:")
    print(df.describe())
    print("\nKiểm tra dữ liệu thiếu:")
    print(df.isnull().sum())

# Train RNN model
def train_rnn(X_train, y_train, input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128))
    model.add(SimpleRNN(64))
    model.add(Dense(1, activation='sigmoid'))  # Or 'softmax' for multi-class
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)

# Choose model based on user input
def suggest_and_train_model(df, model_type):
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if model_type == 'classification':
        print(f"Đề xuất các model Classification: RandomForest, SVM, LogisticRegression")
        model_choice = input("Chọn model bạn muốn (RandomForest, SVM, LogisticRegression): ").lower()
        if model_choice == 'randomforest':
            model = RandomForestClassifier()
        elif model_choice == 'svm':
            model = SVC()
        elif model_choice == 'logisticregression':
            model = LogisticRegression()
        else:
            print("Model không hợp lệ!")
            return
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Độ chính xác của model {model_choice}: {accuracy:.2f}")

    elif model_type == 'regression':
        print(f"Đề xuất các model Regression: LinearRegression, SVR, RandomForestRegressor")
        model_choice = input("Chọn model bạn muốn (LinearRegression, SVR, RandomForestRegressor): ").lower()
        if model_choice == 'linearregression':
            model = LinearRegression()
        elif model_choice == 'svr':
            model = SVR()
        elif model_choice == 'randomforestregressor':
            model = RandomForestRegressor()
        else:
            print("Model không hợp lệ!")
            return
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error của model {model_choice}: {mse:.2f}")

    elif model_type == 'nlp':
        print(f"Đề xuất model NLP: RNN")
        model_choice = input("Chọn model NLP bạn muốn (RNN): ").lower()
        if model_choice == 'rnn':
            # Reshape data for RNN
            X_train_rnn = X_train.values.reshape(-1, X_train.shape[1], 1)  # Reshape to 3D for RNN
            train_rnn(X_train_rnn, y_train.values, input_dim=X.shape[1])  # Ensure y_train is an array
        else:
            print("Model không hợp lệ!")

    elif model_type == 'llm':
        print(f"Đề xuất model LLM: LLaMA, Falcon")
        model_choice = input("Chọn model LLM bạn muốn (LLaMA, Falcon): ").lower()
        if model_choice == 'llama':
            # Initialize LLaMA model
            tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')
            model = LlamaForCausalLM.from_pretrained('huggyllama/llama-7b')
        elif model_choice == 'falcon':
            # Initialize Falcon model
            tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b')
            model = FalconForCausalLM.from_pretrained('tiiuae/falcon-7b')
        else:
            print("Model không hợp lệ!")
            return

        print(f"Model LLM được chọn: {model_choice}. Tạm thời không triển khai model này trong ví dụ này.")

    else:
        print("Loại model không hợp lệ!")

# Main function to run the tool
def run_tool():
    from google.colab import files
    uploaded = files.upload()  # Upload the file

    for filename in uploaded.keys():
        file_path = filename
        break

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            print("Chỉ hỗ trợ file CSV và Excel!")
            return
    except Exception as e:
        print(f"Lỗi khi đọc dataset: {e}")
        return

    analyze_data(df)
    visualize_data(df)

    model_type = input("Nhập loại mô hình bạn muốn sử dụng (Classification, Regression, NLP, LLM): ").lower()
    suggest_and_train_model(df, model_type)

# Run the tool
if __name__ == "__main__":
    run_tool()
