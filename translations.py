"""
Translations for GGUF Converter
Supports: Russian (ru), English (en), Chinese (zh)
"""

TRANSLATIONS = {
    "ru": {
        # Window
        "window_title": "GGUF Конвертер",
        "version_text": "GGUF конвертер от miha2017. вер. 1.9",
        
        # Header
        "title": "GGUF Конвертер",
        
        # Model selection
        "model_from_downloads": "Модель из Загрузок:",
        "refresh": "Обновить",
        "browse": "Обзор",
        
        # Model info
        "type": "Тип",
        "size": "Размер",
        "dtype": "Тип данных",
        "tensors": "Тензоры",
        "output": "Выход",
        
        # Quantization
        "quantization_level": "Уровень квантизации:",
        "recommended": "рекомендуется",
        "near_lossless": "почти без потерь",
        
        # Status
        "status": "Статус:",
        "ready": "Готов",
        "preparing": "Подготовка...",
        "loading_weights": "Загрузка весов...",
        "quantizing": "Квантизация...",
        "processing": "Обработка",
        "done": "Готово!",
        "conversion_complete": "Конвертация завершена!",
        "conversion_failed": "Ошибка конвертации",
        "cancelling": "Отмена...",
        
        # AWQ Support
        "awq_detected": "Обнаружена AWQ модель. Требуется деквантизация перед конвертацией.",
        "awq_dequantizing": "Деквантизация AWQ модели в FP16...",
        "awq_dequantized_layers": "Деквантизировано слоёв",
        "awq_memory_warning": "Внимание: деквантизация AWQ требует ~{0} ГБ памяти",
        "awq_processing": "Обработка AWQ слоя",
        
        # Log
        "log": "Лог:",
        "time": "Время",
        "found_models": "Найдено моделей в Загрузках",
        "analyzing": "Анализ",
        "model_name_from_config": "Имя модели из config.json",
        "model_name_from_metadata": "Имя модели из метаданных",
        "model_name_from_folder": "Имя модели из папки",
        "model_name_from_filename": "Имя модели из файла",
        "playing": "Воспроизведение",
        "no_music_files": "Нет музыкальных файлов в папке music/",
        "music_error": "Ошибка музыки",
        "created_music_folder": "Создана папка music",
        
        # Buttons
        "convert": "Конвертировать",
        "cancel": "Отмена",
        
        # Dialogs
        "error": "Ошибка",
        "info": "Информация",
        "warning": "Предупреждение",
        "select_model_first": "Сначала выберите модель",
        "already_gguf": "Уже в формате GGUF",
        "success": "Успешно!",
        "file": "Файл",
        "compression": "Сжатие",
        "ok": "OK",
        
        # Single instance warning
        "already_running": "GGUF Конвертер уже запущен!",
        "close_previous": "Закройте предыдущий экземпляр программы.",
        
        # Languages
        "lang_ru": "RU",
        "lang_en": "EN", 
        "lang_zh": "中文",
        
        # New features
        "output_folder": "Папка вывода:",
        "select_folder": "Выбрать",
        "batch_mode": "Пакетный режим",
        "inspect_gguf": "Инспекция GGUF",
        "select_files": "Выбрать файлы",
        "files_selected": "Выбрано файлов",
        "batch_progress": "Обработка файла",
        "batch_complete": "Пакетная конвертация завершена",
        "batch_results": "Результаты",
        "successful": "Успешно",
        "failed": "Ошибка",
        "inspect_title": "Инспекция GGUF файла",
        "select_gguf": "Выберите GGUF файл",
        "gguf_info": "Информация о GGUF",
        "tensor_list": "Список тензоров",
        "tensor_name": "Имя",
        "tensor_shape": "Форма",
        "tensor_dtype": "Тип",
        "tensor_size": "Размер",
        "total_tensors": "Всего тензоров",
        "total_size": "Общий размер",
        "copy_info": "Копировать",
        "export_csv": "Экспорт CSV",
        "no_gguf_selected": "Выберите GGUF файл для инспекции",
        "invalid_gguf": "Некорректный GGUF файл",
        "gguf_version": "Версия GGUF",
        "metadata": "Метаданные",
    },
    
    "en": {
        # Window
        "window_title": "GGUF Converter",
        "version_text": "GGUF converter by miha2017. ver. 1.9",
        
        # Header
        "title": "GGUF Converter",
        
        # Model selection
        "model_from_downloads": "Model from Downloads:",
        "refresh": "Refresh",
        "browse": "Browse",
        
        # Model info
        "type": "Type",
        "size": "Size",
        "dtype": "Dtype",
        "tensors": "Tensors",
        "output": "Output",
        
        # Quantization
        "quantization_level": "Quantization level:",
        "recommended": "recommended",
        "near_lossless": "near-lossless",
        
        # Status
        "status": "Status:",
        "ready": "Ready",
        "preparing": "Preparing...",
        "loading_weights": "Loading weights...",
        "quantizing": "Quantizing...",
        "processing": "Processing",
        "done": "Done!",
        "conversion_complete": "Conversion complete!",
        "conversion_failed": "Conversion failed",
        "cancelling": "Cancelling...",
        
        # AWQ Support
        "awq_detected": "AWQ model detected. Dequantization required before conversion.",
        "awq_dequantizing": "Dequantizing AWQ model to FP16...",
        "awq_dequantized_layers": "Dequantized layers",
        "awq_memory_warning": "Warning: AWQ dequantization requires ~{0} GB memory",
        "awq_processing": "Processing AWQ layer",
        
        # Log
        "log": "Log:",
        "time": "Time",
        "found_models": "Found models in Downloads",
        "analyzing": "Analyzing",
        "model_name_from_config": "Model name from config.json",
        "model_name_from_metadata": "Model name from metadata",
        "model_name_from_folder": "Model name from folder",
        "model_name_from_filename": "Model name from filename",
        "playing": "Playing",
        "no_music_files": "No music files in music/ folder",
        "music_error": "Music error",
        "created_music_folder": "Created music folder",
        
        # Buttons
        "convert": "Convert",
        "cancel": "Cancel",
        
        # Dialogs
        "error": "Error",
        "info": "Info",
        "warning": "Warning",
        "select_model_first": "Select a model first",
        "already_gguf": "Already GGUF format",
        "success": "Success!",
        "file": "File",
        "compression": "Compression",
        "ok": "OK",
        
        # Single instance warning
        "already_running": "GGUF Converter is already running!",
        "close_previous": "Close the previous instance.",
        
        # Languages
        "lang_ru": "RU",
        "lang_en": "EN",
        "lang_zh": "中文",
        
        # New features
        "output_folder": "Output folder:",
        "select_folder": "Select",
        "batch_mode": "Batch mode",
        "inspect_gguf": "Inspect GGUF",
        "select_files": "Select files",
        "files_selected": "Files selected",
        "batch_progress": "Processing file",
        "batch_complete": "Batch conversion complete",
        "batch_results": "Results",
        "successful": "Successful",
        "failed": "Failed",
        "inspect_title": "GGUF File Inspector",
        "select_gguf": "Select GGUF file",
        "gguf_info": "GGUF Information",
        "tensor_list": "Tensor List",
        "tensor_name": "Name",
        "tensor_shape": "Shape",
        "tensor_dtype": "Type",
        "tensor_size": "Size",
        "total_tensors": "Total tensors",
        "total_size": "Total size",
        "copy_info": "Copy",
        "export_csv": "Export CSV",
        "no_gguf_selected": "Select a GGUF file to inspect",
        "invalid_gguf": "Invalid GGUF file",
        "gguf_version": "GGUF Version",
        "metadata": "Metadata",
    },
    
    "zh": {
        # Window
        "window_title": "GGUF 转换器",
        "version_text": "GGUF 转换器 by miha2017. 版本 1.9",
        
        # Header
        "title": "GGUF 转换器",
        
        # Model selection
        "model_from_downloads": "下载文件夹中的模型:",
        "refresh": "刷新",
        "browse": "浏览",
        
        # Model info
        "type": "类型",
        "size": "大小",
        "dtype": "数据类型",
        "tensors": "张量",
        "output": "输出",
        
        # Quantization
        "quantization_level": "量化级别:",
        "recommended": "推荐",
        "near_lossless": "近乎无损",
        
        # Status
        "status": "状态:",
        "ready": "就绪",
        "preparing": "准备中...",
        "loading_weights": "加载权重...",
        "quantizing": "量化中...",
        "processing": "处理中",
        "done": "完成!",
        "conversion_complete": "转换完成!",
        "conversion_failed": "转换失败",
        "cancelling": "取消中...",
        
        # AWQ Support
        "awq_detected": "检测到AWQ模型。转换前需要反量化。",
        "awq_dequantizing": "正在将AWQ模型反量化为FP16...",
        "awq_dequantized_layers": "已反量化层数",
        "awq_memory_warning": "警告：AWQ反量化需要约{0} GB内存",
        "awq_processing": "正在处理AWQ层",
        
        # Log
        "log": "日志:",
        "time": "时间",
        "found_models": "在下载文件夹中找到模型",
        "analyzing": "分析中",
        "model_name_from_config": "从 config.json 获取模型名称",
        "model_name_from_metadata": "从元数据获取模型名称",
        "model_name_from_folder": "从文件夹获取模型名称",
        "model_name_from_filename": "从文件名获取模型名称",
        "playing": "播放中",
        "no_music_files": "music/ 文件夹中没有音乐文件",
        "music_error": "音乐错误",
        "created_music_folder": "已创建 music 文件夹",
        
        # Buttons
        "convert": "转换",
        "cancel": "取消",
        
        # Dialogs
        "error": "错误",
        "info": "信息",
        "warning": "警告",
        "select_model_first": "请先选择模型",
        "already_gguf": "已经是 GGUF 格式",
        "success": "成功!",
        "file": "文件",
        "compression": "压缩",
        "ok": "确定",
        
        # Single instance warning
        "already_running": "GGUF 转换器已在运行!",
        "close_previous": "请关闭之前的实例。",
        
        # Languages
        "lang_ru": "RU",
        "lang_en": "EN",
        "lang_zh": "中文",
        
        # New features
        "output_folder": "输出文件夹:",
        "select_folder": "选择",
        "batch_mode": "批量模式",
        "inspect_gguf": "检查 GGUF",
        "select_files": "选择文件",
        "files_selected": "已选择文件",
        "batch_progress": "正在处理文件",
        "batch_complete": "批量转换完成",
        "batch_results": "结果",
        "successful": "成功",
        "failed": "失败",
        "inspect_title": "GGUF 文件检查器",
        "select_gguf": "选择 GGUF 文件",
        "gguf_info": "GGUF 信息",
        "tensor_list": "张量列表",
        "tensor_name": "名称",
        "tensor_shape": "形状",
        "tensor_dtype": "类型",
        "tensor_size": "大小",
        "total_tensors": "张量总数",
        "total_size": "总大小",
        "copy_info": "复制",
        "export_csv": "导出 CSV",
        "no_gguf_selected": "请选择要检查的 GGUF 文件",
        "invalid_gguf": "无效的 GGUF 文件",
        "gguf_version": "GGUF 版本",
        "metadata": "元数据",
    },
}


def get_text(lang: str, key: str) -> str:
    """Get translated text for a key."""
    if lang not in TRANSLATIONS:
        lang = "ru"
    return TRANSLATIONS[lang].get(key, key)
