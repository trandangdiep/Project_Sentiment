import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import dill
import re
from scipy.sparse import hstack
from wordcloud import WordCloud
from collections import Counter
from processor.text import TextProcessor
text_processor = TextProcessor()

# read data
data = pd.read_csv('data/output.csv',encoding='utf-8')
data1 = pd.read_csv('data/Danh_gia.csv',encoding='utf-8')
merged_data = pd.read_csv('data/merged_data.csv',encoding='utf-8')
merged_data = merged_data.dropna()

# Load files
#LOAD POSITIVE_VN
file = open('files/positive_VN.txt', 'r', encoding="utf8")
positive_VN_lst = file.read().split('\n')
file.close()
#-------------------------------------------------
#LOAD NEGATIVE_VN
file = open('files/negative_VN.txt', 'r', encoding="utf8")
negative_VN_lst = file.read().split('\n')
file.close()
#-------------------------------------------------
#LOAD POSITIVE_EMOjiS
file = open('files/positive_emojis.txt', 'r', encoding="utf8")
positive_emojis_lst = file.read().split('\n')
file.close()
#------------------------------------------------
#LOAD NEGATIVE_EMOJIS
file = open('files/negative_emojis.txt', 'r', encoding="utf8")
negative_emojis_lst = file.read().split('\n')
file.close()
#LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()


#Load models
## model tfidft
with open('models/model_Tfidf.pkl', 'rb') as file:  
    model_tfidf = pickle.load(file)
## model RandomForest
with open('models/model_rf.pkl', 'rb') as f:
    model_rf = pickle.load(f)

model = model_rf['model']
metrics = model_rf['metrics']

## chuẩn hóa các từ lặp
with open('Process_text_ttth/normalize_repeated_characters.pkl', 'rb') as f:
    normalize_repeated_characters = dill.load(f)
## ráp các từ đặc biệt
with open('Process_text_ttth/process_special_word.pkl', 'rb') as f:
    process_special_word = dill.load(f)
## hàm xử lý test
with open('Process_text_ttth/process_text.pkl', 'rb') as f:
    process_text = dill.load(f)
## đếm các cảm xúc
with open('Process_text_ttth/process_sentiment.pkl', 'rb') as f:
    process_sentiment = dill.load(f)
## đếm từ positive , negative
with open('Process_text_ttth/words_count.pkl', 'rb') as f:
    words_count = dill.load(f)

# hàm chuyển data sang dataframe
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


    # GUI
st.title("Data Science Project")
st.write("## Sentiment Analysis")

menu = ["Business Objective", "Build Project", "New Prediction","Product Search"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:\n
                Trần Đăng Diệp """)
st.sidebar.write("""#### Giảng viên hướng dẫn: 
                        """)
st.sidebar.write("""#### Thời gian thực hiện: 4/6/2025""")
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Vấn đề : Để thu hút được một lượng khách lớn và ổn định , trước hết chúng ta phải đáp ứng được các nhu cầu của khách hàng, vậy vấn đề ở đây là chúng ta phải hiểu được khách hàng , từ đó cải thiện chất lượng sản phẩm cũng như các dịch vụ đi kèm.
    """)  
    st.write("""###### => Yêu cầu: Xây dựng thuật toán phân tích cảm xúc thông qua đánh giá của khách hàng về sản phẩm.""")
    st.image("images/header.png",width=900)
    

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data")
    st.dataframe(data1[['noi_dung_binh_luan']].head(3))
    st.dataframe(data[['noi_dung_binh_luan','length','sentiment','positive_count','negative_count']].head(3)) 

    st.write("##### 2. Visualize sentiment")
    fig1 = sns.countplot(data=data[['sentiment']], x='sentiment')    
    st.pyplot(fig1.figure)
    st.write("Dữ liệu bị mất cân bằng")
    st.write("Biểu sau y_train và y_test sau khi được chia theo lỷ lệ 8/2 và cân bằng dữ liệu")
    st.image("images/sentiment_balanced.png",width=1200)

    st.write("##### 3. Build model...")
    st.write("Xây dựng bằng Mô hình Random Forest")
    st.image("images/train_rf1.png",width=1200)

    st.write("##### 4. Evaluation")
    st.write("Time train : 7.15s với tập dữ liệu 16539 dòng")
    st.write("Time test : 0.17s với tập dữ liệu 4135 dòng")
    st.code("Score train:"+ str(round(metrics['score_train'],2)) + " vs Score test:" + str(round(metrics['score_test'],2)))
    st.code("Accuracy:"+str(round(metrics['accuracy'],2)))
    st.write("###### Confusion matrix:")
    st.code(metrics['confusion_matrix'])
    st.write("###### Classification report:")
    st.code(metrics['classification_report'])
    st.code("Roc AUC score:" + str(round(metrics['roc_auc'],2)))

    # calculate roc curve
    st.write("###### ROC curve")
    st.image('images/roc_rf.png',width=900)

    st.write("##### 5. Summary: This model is good enough for Sentiment Analysis.")

elif choice == 'New Prediction':
    st.subheader("Select data")
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input","upload , Predict by product code"))
    flag = False
    if type=="Upload":
        # Upload file
        st.write('Only supports csv and txt files')
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None,names=['noi_dung_binh_luan'])
            st.dataframe(lines)                  
            st.write("Content:")
            if len(lines['noi_dung_binh_luan'][0])>0:
                st.code(lines)

                # lấy độ dài chuỗi
                lines['length'] = lines['noi_dung_binh_luan'].map(lambda x : len(x))

                # Xử lý text
                process_text(lines, emoji_dict, teen_dict, wrong_lst, stopwords_lst, english_dict)
                process_sentiment(lines, words_count,positive_VN_lst,negative_VN_lst)

                # chuẩn text
                x_new = model_tfidf.transform(lines['noi_dung_binh_luan'])
                X_final = hstack([x_new, lines[['length','positive_count','negative_count']]]) 

                # Dự đoán
                st.write("After Predict : ")
                st.write("New predictions (1: positive, 0: negative): ")      
                y_pred_new = model.predict(X_final) 
                lines['Predict'] = y_pred_new
                st.dataframe(lines)

                #Download
                csv = convert_df(lines)
                st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name="output.csv",
                mime="csv"
                )           

    if type=="Input":        
        content = st.text_area(label="Input your content:")
        if content!="":
            line = np.array([content])

            st.write("Content:")
            if len(line)>0:
                st.code(line)

                # tạo dataframe
                lines = pd.DataFrame(line,columns=['noi_dung_binh_luan'])

                # lấy độ dài chuỗi
                lines['length'] = lines['noi_dung_binh_luan'].map(lambda x : len(x))

                # xử lý text
                process_text(lines, emoji_dict, teen_dict, wrong_lst, stopwords_lst, english_dict)
                process_sentiment(lines, words_count,positive_VN_lst,negative_VN_lst)
                x_new = model_tfidf.transform(lines['noi_dung_binh_luan'])        
                X_final = hstack([x_new, lines[['length','positive_count','negative_count']]]) 

                # dự đoán
                st.write("After Predict : ")
                st.write("New predictions (1: positive, 0: negative): ")       
                y_pred_new = model.predict(X_final)          
                lines['Predict'] = y_pred_new
                st.dataframe(lines)

    if type=="upload , Predict by product code":
        # Upload file
        st.write('Only supports csv and txt files , must have column ma_san_pham')
        uploaded_file_1 = st.file_uploader("Choose a file",type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1,header=None,names=['noi_dung_binh_luan','ma_san_pham'],)                  
            
            # lấy random sản phẩm
            st.session_state.random_products = lines

            # Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
            if 'selected_ma_san_pham' not in st.session_state:
                # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
                st.session_state.selected_ma_san_pham = None

            # Theo cách cho người dùng chọn sản phẩm từ dropdown
            # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
            product_options = [(row['noi_dung_binh_luan'], row['ma_san_pham']) for index, row in st.session_state.random_products.iterrows()]
            st.session_state.random_products
            # Tạo một dropdown với options là các tuple này
            selected_product = st.selectbox(
                "Chọn sản phẩm",
                options=product_options,
                format_func=lambda x: x[1]
            )
            # Display the selected product
            st.write("Bạn đã chọn:", selected_product)

            # Cập nhật session_state dựa trên lựa chọn hiện tại
            st.session_state.selected_ma_san_pham = selected_product[1]

            if st.session_state.selected_ma_san_pham:
                st.write("ma_san_pham: ", st.session_state.selected_ma_san_pham)
                # Hiển thị thông tin sản phẩm được chọn
                selected_product = lines[lines['ma_san_pham'] == st.session_state.selected_ma_san_pham]

                if not selected_product.empty:
                    st.write('#### Bạn vừa chọn:')
                    st.write('### ', selected_product)
                    # lấy độ dài chuỗi
                    selected_product['length'] = selected_product['noi_dung_binh_luan'].map(lambda x : len(x))

                    # Xử lý text
                    process_text(selected_product, emoji_dict, teen_dict, wrong_lst, stopwords_lst, english_dict)
                    process_sentiment(selected_product, words_count,positive_VN_lst,negative_VN_lst)

                    # chuẩn text
                    x_new = model_tfidf.transform(selected_product['noi_dung_binh_luan'])
                    X_final = hstack([x_new, selected_product[['length','positive_count','negative_count']]]) 

                    # Dự đoán
                    st.write("After Predict : ")
                    st.write("New predictions (1: positive, 0: negative): ")       
                    y_pred_new = model.predict(X_final)       
                    selected_product['Predict'] = y_pred_new
                    st.dataframe(selected_product)

                    #download
                    csv = convert_df(selected_product) 
                    st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name="output.csv",
                    mime="csv"
                    )   

                else:
                    st.write(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")

elif choice == 'Product Search':
    # lấy random sản phẩm
    st.session_state.random_products = merged_data[['ten_san_pham','ma_san_pham']]
    # Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
    if 'selected_ma_san_pham' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
        st.session_state.selected_ma_san_pham = None

    # Theo cách cho người dùng chọn sản phẩm từ dropdown
    # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    product_options = [( row['ten_san_pham'],row['ma_san_pham']) for index, row in st.session_state.random_products.drop_duplicates().iterrows()]
    st.dataframe(merged_data)
    # Tạo một dropdown với options là các tuple này
    selected_product = st.selectbox(
        "Chọn sản phẩm",
        options=product_options,
        format_func=lambda x: x[0]
    )
    # Display the selected product
    st.write("Bạn đã chọn:", selected_product)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_ma_san_pham = selected_product[1]

    if st.session_state.selected_ma_san_pham:

        st.write('### information :',selected_product[0])
        # Hiển thị thông tin sản phẩm được chọn
        selected_product = merged_data[merged_data['ma_san_pham'] == st.session_state.selected_ma_san_pham]

        if not selected_product.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_product)
            st.write('### WordCloud for :',", ".join(selected_product["ten_san_pham"].drop_duplicates().tolist()))
            for i in range(0,2):
                if i == 0:
                    a = 'positive_count'
                    b = 'negative_count'
                    c = 'Positive'
                    
                else:
                    a = 'negative_count'
                    b = 'positive_count'
                    c = 'Negative'
                st.write("### WordCloud for ",c)
                product_reviews = selected_product[(selected_product[f'{a}'] != 0) & (selected_product[f'{b}'] == 0)]
                if len(product_reviews) > 0 :
                    text = ' '.join(product_reviews['noi_dung_binh_luan'].astype(str))
                    # Lấy 10 từ phổ biến nhất
                    words = ' '.join(product_reviews['noi_dung_binh_luan'].astype(str)).split()
                    word_freq = Counter(words)
                    common_words = pd.DataFrame(word_freq.most_common(40), columns=['Word', 'Frequency'])
                    st.dataframe(common_words.head(10))
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                    # Vẽ biểu đồ
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.set_title(f"WordCloud for {c}")
                    ax.axis("off")
                    # Hiển thị WordCloud trên Streamlit
                    st.pyplot(fig)
                else :
                    st.write("### không có từ ",c)



            # Trực quan hóa bình luận tích cực và tiêu cực
            st.write("### Trực quan hóa bình luận tích cực và tiêu cực")
            sentiment_counts = selected_product[['positive_count', 'negative_count']].fillna(0).astype(int).sum()
            st.dataframe(sentiment_counts)
            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(10, 5))
            sentiment_counts.plot(kind="bar", color=["green", "red"], ax=ax)
            name = ", ".join(selected_product["ten_san_pham"].drop_duplicates().tolist())
            ax.set_title(f"Sentiment Counts for {name}")
            ax.set_ylabel("Count")
            ax.set_xticklabels(["Positive", "Negative"], rotation=0)
            # Hiển thị biểu đồ trong Streamlit
            st.pyplot(fig)
           

            # Trực quan hóa tỷ lệ bình luận tích cực và tiêu cực
            labels = ['Positive', 'Negative']
            sizes = [sentiment_counts['positive_count'], sentiment_counts['negative_count']]
            colors = ['#66b3ff', '#ff9999']
            # Vẽ biểu đồ tròn
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(
                sizes, 
                labels=labels, 
                colors=colors, 
                autopct='%1.1f%%', 
                startangle=90
            )
            ax.set_title("Tỷ lệ đánh giá tích cực và tiêu cực")
            ax.axis('equal')  # Đảm bảo hình tròn
            # Hiển thị biểu đồ trong Streamlit
            st.pyplot(fig)

            st.write("### Tổng số sao của sản phẩm ",", ".join(selected_product["ten_san_pham"].drop_duplicates().tolist()))
            fig, ax = plt.subplots(figsize=(10, 6))
            (
                selected_product["so_sao"]
                .value_counts()
                .sort_index()
                .plot(kind="bar", ax=ax)
            )
            ax.set_title("Phân phối số sao")
            ax.set_xlabel("Số sao")
            ax.set_ylabel("Số lượng đánh giá")
            ax.set_xticks(range(len(selected_product["so_sao"].unique())))
            ax.set_xticklabels(selected_product["so_sao"].unique())
            # Hiển thị biểu đồ trên Streamlit
            st.pyplot(fig)
        else:
            st.write(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")              
            
        
                
         
             
           


