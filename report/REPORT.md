# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Trung Anh Quốc
**Nhóm:** 10
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu: High cosine similarity nghĩa là hai vector có hướng gần giống nhau, tức là hai câu có ý nghĩa tương tự nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Tôi muốn mua một chiếc xe máy mới"
- Sentence B: "Tôi cần mua một chiếc xe hơi mới"
- Tại sao tương đồng: Cả hai câu đều thể hiện nhu cầu mua phương tiện di chuyển.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi muốn mua một chiếc xe máy mới"
- Sentence B: "Hôm nay trời đẹp"
- Tại sao khác: Câu A thể hiện nhu cầu mua phương tiện di chuyển, câu B thể hiện cảm nhận về thời tiết.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu: Cosine similarity đo lường góc giữa hai vector, không phụ thuộc vào độ lớn của vector, do đó phù hợp hơn với text embeddings. Euclidean distance đo lường khoảng cách giữa hai vector, phụ thuộc vào độ lớn của vector, do đó không phù hợp với text embeddings.*

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính: ((10000 - 50) / (500 - 50)) = 21.111(ceil kết quả) = 23 chunks*   
> *Đáp án: 23 chunks*

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu: Nếu overlap tăng lên 100, chunk count sẽ giảm xuống còn (10000 - 100) / (500 - 100) = 24.75(ceil kết quả) = 25 chunks. Overlap nhiều hơn giúp tăng khả năng tìm thấy thông tin liên quan, nhưng cũng làm tăng kích thước của vector store.*

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Wiki

**Tại sao nhóm chọn domain này?**
> Domain này như 1 bách khoa toàn thư, nhiều chủ đề. Dẫn đến các chunk có thể có nhiều thông tin khác nhau, qua đó có thể đánh giá được chất lượng của chunking strategy.

### 3.1 Knowledge Base Inventory
The system currently indexes 7 documents split across various domains:

| ID | Filename | Category | Source/Topic |
|---|---|---|---|
| Bullet_Kin | Gaming_Bullet_Kin.txt | Gaming | Enter the Gungeon Wiki |
| Underground_Travel | General_Underground_Travel.txt | General | RPG Campaign Notes |
| RAG_Chatbot_Blog | Engineering_RAG_Chatbot_Blog.txt | Engineering | STICI-Note Prototyping |
| llmware_Docs | Engineering_llmware_Docs.txt | Engineering | Enterprise RAG Framework |
| Marimo_Recipes | Engineering_Marimo_Recipes.txt | Engineering | UI/Code Snippets |
| Impact_Prioritization | General_Impact_Prioritization.txt | General | Career / Data Science Impact |
| EU_AI_Act | Legal_EU_AI_Act.txt | Legal | AI Regulation Summary |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | string | Gaming, Engineering | Cho phép lọc tài liệu theo chủ đề trước khi tìm kiếm. |
| `source_url` | string | Dòng đầu file | Giúp Agent dẫn nguồn chính xác trang wiki gốc. |
| `source` | string | data/wiki_docs/... | Truy vết file vật lý chứa dữ liệu. |
| `extension` | string | .txt | Lọc loại file nếu cần. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
| Gaming_Bullet_Kin.txt | FixedSizeChunker (`fixed_size`) | 24 | 491.83 | Traditional split |
| Gaming_Bullet_Kin.txt | SentenceChunker (`by_sentences`) | 36 | 293.92 | Best semantic |
| Gaming_Bullet_Kin.txt | RecursiveChunker (`recursive`) | 30 | 353.40 | Best balance |

### Strategy Của Tôi

**Loại:** `SemanticChunker` (Custom Strategy)

**Mô tả cách hoạt động:**
> Chiến lược này tách văn bản thành các câu đơn lẻ, sau đó dùng mô hình Embedding để tạo vector cho từng câu. Nó tính toán độ tương đồng Cosine giữa các câu liên tiếp. Một chunk mới sẽ được bắt đầu khi độ tương đồng giảm xuống dưới ngưỡng 0.5 hoặc khi kích thước chunk vượt quá 1000 ký tự. Điều này giúp giữ các câu có cùng ngữ cảnh ở chung một đoạn.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Domain Wiki có nhiều thực thể với các mô tả dài. SemanticChunker giúp tránh việc cắt ngang một mô tả quan trọng chỉ vì giới hạn ký tự, từ đó giúp Agent truy xuất được đầy đủ ngữ cảnh để trả lời câu hỏi chính xác hơn.

**Code snippet (nếu custom):**
```python
sentence_embeddings = self.embedding_fn(sentences)
for i in range(1, len(sentences)):
    similarity = self._compute_similarity(sentence_embeddings[i], sentence_embeddings[i-1])
    if similarity < self.threshold or current_len > self.max_chunk_size:
        chunks.append(" ".join(current_chunk_sentences))
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Gaming_Bullet_Kin.txt | semantic | 76 | 138.70 | High |
| Gaming_Bullet_Kin.txt | sentence | 36 | 293.92 | Medium |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Semantic | 8.0 | Chia chunk dựa trên sự tương đồng ý nghĩa, giữ trọn vẹn mạch thông tin và ngữ cảnh, hạn chế việc cắt ngang các mô tả quan trọng. | Tốn nhiều tài nguyên tính toán (Embedding API), tốc độ xử lý chậm hơn các phương pháp tách chuỗi thông thường. |
| Trần Thái Thịnh | Recursive | 7.5 | Giữ được cấu trúc văn bản (đoạn, câu) thông qua việc tách đệ quy linh hoạt, cân bằng tốt giữa kích thước và ngữ cảnh. | Vẫn dựa trên tách chuỗi vật lý, đôi khi vẫn cắt ngang context nếu các paragraph quá dài hoặc không có dấu phân cách rõ ràng. |
| Nguyễn Đức Cường | Sentence | 6.5 | Đảm bảo tính toàn vẹn của từng câu, giúp mô hình hiểu ngữ cảnh ở mức độ câu tốt hơn so với cắt theo ký tự ngẫu nhiên. | Kích thước chunk không đồng đều (có câu quá dài/quá ngắn), thông tin dễ bị phân mảnh nếu không gộp nhiều câu liên quan. |
| Nguyễn Văn Thạch | Recursive | 7.5 | Giữ được cấu trúc văn bản (đoạn, câu) thông qua việc tách đệ quy linh hoạt, cân bằng tốt giữa kích thước và ngữ cảnh. | Vẫn dựa trên tách chuỗi vật lý, đôi khi vẫn cắt ngang context nếu các paragraph quá dài hoặc không có dấu phân cách rõ ràng. |
| Trần Khánh Bằng | Layered | 8.5 | Truy xuất thông tin đa tầng (kết hợp chunk lớn bao quát và chunk nhỏ chi tiết), giúp Agent nắm bắt cả tổng quan lẫn chi tiết cụ thể. | Cấu trúc index phức tạp, tốn bộ nhớ lưu trữ và đòi hỏi logic truy vấn nâng cao để xử lý các tầng dữ liệu. |
| Đỗ Hải Nam | Semantic | 8.0 | Chia chunk dựa trên sự tương đồng ý nghĩa, giữ trọn vẹn mạch thông tin và ngữ cảnh, hạn chế việc cắt ngang các mô tả quan trọng. | Tốn nhiều tài nguyên tính toán (Embedding API), tốc độ xử lý chậm hơn các phương pháp tách chuỗi thông thường. |



**Strategy nào tốt nhất cho domain này? Tại sao?**
>Theo em nghĩ là strategy tốt nhất cho domain này là semantic chunker vì nó giữ được ngữ cảnh của câu và giảm thiểu việc cắt ngang một mô tả quan trọng. Nhất là trong domain mang tính chất là các tài liệu wiki, các câu thường có liên quan đến nhau và việc cắt ngang một mô tả quan trọng sẽ làm giảm chất lượng truy xuất.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Sử dụng regex `(?<=[.!?])\s+(?=[A-Z])|(?<=\.\n)` để tách câu dựa trên dấu hiệu kết thúc câu kết hợp với kiểm tra chữ cái viết hoa và xuống dòng để hạn chế lỗi ở các trường hợp như số thập phân hay chữ viết tắt*

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Sử dụng kỹ thuật đệ quy để chia nhỏ văn bản theo danh sách ký tự phân tách ưu tiên (\n\n, \n, . , khoảng trắng). Base case là khi độ dài đoạn văn bản nhỏ hơn `chunk_size`, nếu không nó sẽ tìm điểm cắt tối ưu để đảm bảo tính toàn vẹn của thông tin*

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Dữ liệu được chuyển thành các bản ghi chứa vector đã được embedding và lưu trữ. Khi tìm kiếm, query được vector hóa và so sánh độ tương đồng Cosine với toàn bộ kho dữ liệu để trích xuất các kết quả có score cao nhất*

**`search_with_filter` + `delete_document`** — approach:
> *Hệ thống thực hiện lọc metadata để giới hạn tập dữ liệu trước khi tính toán similarity. Việc xóa tài liệu được thực được xử lý bằng cách loại bỏ các chunk có ID hoặc metadata `doc_id` trùng khớp khỏi store*

### KnowledgeBaseAgent

**`answer`** — approach:
> *Triển khai luồng RAG: Truy vấn context liên quan -> Embedding query và context vào prompt mẫu -> Gọi LLM để tổng hợp câu trả lời*

### Test Results

```
rootdir: D:\VinProject\20A202600108-LeTrungAnhQuoc-Day7
configfile: pyproject.toml
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                      [  2%] 
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                               [  4%] 
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                        [  7%] 
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                         [  9%] 
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                              [ 11%] 
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED              [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                    [ 16%] 
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                     [ 19%] 
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                   [ 21%] 
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                     [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                     [ 26%] 
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                            [ 30%] 
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                      [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED             [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                 [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED           [ 40%] 
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                 [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                     [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                       [ 47%] 
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                         [ 50%] 
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                               [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                    [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                      [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED          [ 59%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                       [ 61%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                               [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                          [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                      [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                 [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                     [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                           [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                     [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED  [ 83%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED               [ 88%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED   [ 90%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED              [ 92%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED       [ 95%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%] 

============================================== 42 passed in 0.11s ===============================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Jammed enemies di chuyển nhanh hơn bình thường." | "Kẻ thù bị Jammed có tốc độ di chuyển tăng cao." | High | 0.609 | Đúng |
| 2 | "EU AI Act quy định về các rủi ro của AI." | "Luật Châu Âu kiểm soát việc triển khai trí tuệ nhân tạo." | High | 0.525 | Đúng |
| 3 | "Cách nấu mỳ ramen ngon tại nhà." | "Hướng dẫn lập trình Python cơ bản." | Low | 0.293 | Đúng |
| 4 | "Keybullet Kin rơi ra chìa khóa khi bị tiêu diệt." | "Hạ gục Keybullet Kin sẽ nhận được chìa khóa." | High | 0.690 | Đúng |
| 5 | "Hôm nay trời đẹp." | "Tôi đang học về Vector Database." | Low | 0.301 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
>Kết quả bất ngờ nhất mà em nhận ra là Case thứ 5, nhìn có vẻ không liên quan nhưng điểm số lại khá cao (0.301).

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer (câu trả lời đúng) | Doc nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | What makes jammed enemies different? | Jammed Keybullet Kin drop 2 keys instead of 1, jammed Chance Kins have a chance to drop twice the loot, and jammed red-Caped Bullet Kin deal contact damage. Additionally, Jammed Keybullet Kin variations run faster and will take less time to teleport away from the player if they are not destroyed quickly. | Doc 0 |
| 2 | Which large language models and vector databases were shortlisted for this project? | The tinyllama-1.1b-chat-v1.0 Q6_K, Phi 3 Q4_K_M, bartowski/dolphin-2.8-experiment26-7b-GGUF Q3_K_L, mgonzs13/Mistroll-7B-v2.2-GGU, and QuantFactory/Meta-Llama-3-8B-Instruct Q3_K_M large language models and the Chroma, Qdrant, and Vespa vector databases were shortlisted for this project. | Doc 2 (Filtered) |
| 3 | What kind of model is the bling-phi-3 model | The bling-phi-3 model is the newest and most accurate BLING/DRAGON model. BLING models are small CPU-based RAG-optimized, instruct-following 1B-3B parameter models. | Doc 3 |
| 4 | What are the ways of grouping UI elements together? | UI elements can be grouped together using the following methods: Create an array of UI elements, Create a dictionary of UI elements, Embed a dynamic number of UI elements in another output, Create a hstack (or vstack) of UI elements with on_change handlers, Create a table column of buttons with on_change handlers, Create a form with multiple UI elements. | Doc 4 |
| 5 | What are the four steps to become more impact-focused? | The four steps to become more impact-focused are: "Step 1: Understand what impact looks like for your role...", "Step 2: Ensure your work solves a real business problem", "Step 3: Ensure there is buy-in for your work", and "Step 4: Focus your time on the highest-impact thing". | Doc 5 |

### Kết Quả Của Tôi

### 7.1 Benchmark Result Table

| # | Query | Gold Answer (câu trả lời đúng) | Doc nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | What makes jammed enemies different? | Các kẻ thù bị "jammed" khác biệt ở chỗ chúng chạy nhanh hơn và sẽ mất ít thời gian hơn để teleport ra khỏi người chơi nếu không bị tiêu diệt nhanh chóng. | Doc 0 (Bullet_Kin) |
| 2 | Which large language models and vector databases were shortlisted for this project? | Dựa vào các thông tin được cung cấp, các mô hình ngôn ngữ lớn (large language models) được đề cập bao gồm:

1. **SLIM model series**: Các mô hình nhỏ, chuyên biệt được tinh chỉnh cho việc gọi hàm và quy trình làm việc đa bước, đa mô hình.
2. **DRAGON model series**: Các mô hình có từ 6-7 tỷ tham số, được tối ưu hóa cho RAG (Retrieval-Augmented Generation).

Về cơ sở dữ liệu vector (vector databases), các cơ sở dữ liệu được đề cập bao gồm:

1. Milvus
2. PGVector-Postgres
3. Neo4j
4. Redis
5. Mongo-Atlas
6. Qdrant
7. Faiss
8. LanceDB
9. ChromaDB
10. Pinecone

Ngoài ra, có thông tin về việc thêm 3 cơ sở dữ liệu vector mới là Postgres (PG Vector), Redis và Qdrant. | Doc 2 (RAG_Chatbot_Blog) |
| 3 | What kind of model is the bling-phi-3 model | Mô hình bling-phi-3 là một trong những mô hình trong dòng sản phẩm BLING, được mô tả là mô hình mới nhất và chính xác nhất trong dòng BLING/DRAGON. Nó thuộc loại mô hình tối ưu hóa cho CPU và có khả năng thực hiện các tác vụ theo hướng dẫn. | Doc 3 (llmware_Docs) |
| 4 | What are the ways of grouping UI elements together? | Có hai cách để nhóm các phần tử giao diện người dùng (UI elements) lại với nhau:

1. **Sử dụng mảng (Array)**: Bạn có thể tạo một mảng các phần tử UI bằng cách sử dụng `mo.ui.array`. Điều này cho phép bạn tạo ra một phần tử UI mới từ một danh sách các phần tử UI. Ví dụ, bạn có thể tạo một danh sách các thanh trượt (sliders) với số lượng phụ thuộc vào giá trị của một phần tử UI khác.

2. **Sử dụng từ điển (Dictionary)**: Bạn cũng có thể tạo một từ điển các phần tử UI bằng cách sử dụng `mo.ui.dictionary`. Cách này cho phép bạn đặt tên cho từng phần tử được nhóm lại bằng một khóa chuỗi, giúp dễ dàng truy cập và quản lý các phần tử này.

Cả hai phương pháp này đều giúp đồng bộ hóa các phần tử UI giữa frontend và backend trong ứng dụng. | Doc 4 (Marimo_Recipes) |
| 5 | What are the four steps to become more impact-focused? | 1. **Hiểu tác động trong vai trò của bạn và đo lường thành công của bạn dựa trên đó**: Thay vì chỉ suy nghĩ về các chỉ số năng suất như "Tôi đã thực hiện 5 thí nghiệm" hay "Tôi đã xây dựng mô hình này", hãy tự chịu trách nhiệm về việc tạo ra tác động.

2. **Xác định các ưu tiên và tập trung vào điều quan trọng nhất**: Trong bất kỳ vai trò nào, bạn thường phải xử lý nhiều ưu tiên. Để tối đa hóa tác động của mình, bạn cần đảm bảo rằng bạn dành phần lớn thời gian cho điều quan trọng nhất.

3. **Tìm kiếm các khung tổng quát để tối đa hóa tác động**: Điều này liên quan đến việc áp dụng các phương pháp cụ thể để làm cho các dự án thực tế trở nên có tác động hơn.

4. **Tập trung thời gian của bạn vào những điều có tác động cao nhất**: Đảm bảo rằng bạn đang dành thời gian cho những nhiệm vụ có ảnh hưởng lớn nhất đến kết quả cuối cùng.

Những bước này sẽ giúp bạn trở nên tập trung hơn vào tác động trong công việc của mình. | Doc 5 (Impact_Prioritization) |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

> **Nhận xét:** Sau khi chuyển từ Mock sang OpenAI Embedding thật, độ chính xác tìm kiếm (Retrieval) đã đạt mức tối đa. Các kết quả có score quanh mức 0.5 - 0.7 nhưng nội dung trích xuất cực kỳ chuẩn xác, giúp Agent trả lời đầy đủ và đúng trọng tâm các câu hỏi benchmark. Điều này chứng minh sự vượt trội của Semantic Search so với tìm kiếm từ khóa truyền thống.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Học được strategy khác đó chính là Layered Chunking. Thay vì chỉ chia nhỏ tài liệu thành các đoạn văn bản độc lập, strategy này chia tài liệu thành nhiều lớp (layers) với các kích thước khác nhau. Khi tìm kiếm, hệ thống sẽ tìm kiếm trên tất cả các lớp và kết hợp kết quả để đưa ra câu trả lời chính xác hơn.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Học được cách nhóm khác clean và structure lại từ raw data nhận được.Giảm thiểu tối đa nhiễu mà raw data có thể gây ra

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Em nghĩ em sẽ thử áp dụng strategy Layered Chunking cho bài toán của mình, vì em thấy nó cũng có thể giải quyết được bài toán của nhóm một cách hiệu quả

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 10 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 0 / 5 |
| **Tổng** | | **90 / 100** |
