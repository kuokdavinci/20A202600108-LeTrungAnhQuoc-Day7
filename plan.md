# Lịch trình Lab Day 7: Embedding & Vector Store

Tài liệu này tóm tắt các bước cần thực hiện để hoàn thành Lab Day 7.

## 🕒 Giai đoạn 1: Cá nhân (Sáng/Giờ đầu)

### 1. Khởi động (Warm-up)
- [x] Trả lời các câu hỏi về **Cosine Similarity** trong `report/REPORT.md`.
- [x] Tính toán số lượng chunk trong bài tập **Chunking Math** và ghi vào report.

### 2. Lập trình cốt lõi (Core Coding)
Hoàn thành các TODO trong thư mục `src/`:
- [x] **src/chunking.py**:
    - [x] `SentenceChunker`: Chia văn bản theo câu.
    - [x] `RecursiveChunker`: Chia văn bản phân cấp (\n\n, \n, . , khoảng trắng).
    - [x] `compute_similarity`: Tính toán Cosine Similarity.
    - [x] `ChunkingStrategyComparator`: Công cụ so sánh các chiến lược.
- [x] **src/store.py**:
    - [x] `EmbeddingStore`: Khởi tạo (Memory/ChromaDB).
    - [x] `add_documents`: Nhúng và lưu trữ.
    - [x] `search`: Tìm kiếm tương đồng.
    - [x] `search_with_filter`: Tìm kiếm kết hợp lọc metadata.
    - [x] `delete_document`: Xóa tài liệu.
- [x] **src/agent.py**:
    - [x] `KnowledgeBaseAgent`: Code vòng lặp RAG (Retrieve -> Prompt -> LLM).

### 3. Kiểm thử & Phân tích
- [x] Chạy lệnh `pytest tests/ -v` để đảm bảo code vượt qua 30+ tests.
- [/] Thực hiện dự đoán Cosine Similarity cho 5 cặp câu và ghi kết quả vào report.

---

## 🕒 Giai đoạn 2: Nhóm (Chiều/Giờ sau)

### 4. Chuẩn bị tài liệu
- [x] Chọn **Domain** (Enter the Gungeon Wiki).
- [x] Thu thập 5-10 tài liệu (.csv files).
- [x] Thiết kế **Metadata Schema** (source_url, index).

### 5. So sánh & Đánh giá
- [ ] Thống nhất **5 Benchmark Queries** kèm đáp án chuẩn (Sử dụng question CSVs).
- [/] Mỗi thành viên thử 1 chiến lược khác nhau (Đang triển khai hỗ trợ CSV).
- [ ] Chạy benchmark và so sánh kết quả retrieval giữa các thành viên.
- [ ] Phân tích trường hợp thất bại (Failure Analysis).

### 6. Hoàn thiện báo cáo
- [ ] Tổng hợp kết quả nhóm vào `report/REPORT.md`.
- [ ] Tự đánh giá điểm số.

---

## 🚀 Lệnh hữu ích
- Cài đặt thư viện: `pip install -r requirements.txt`
- Chạy test: `pytest tests/ -v`
- Chạy demo: `python main.py`
