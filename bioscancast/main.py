# question = ForecastQuestion(
#     id="Q123",
#     text="Will country X report more than 50 confirmed human cases of pathogen Y by 30 June 2026?",
#     created_at=now()
# )

# search_results = run_search_stage(question, config)

# filtered_docs = run_filtering_stage(question, search_results, config)

# downstream
# extracted_texts = run_extraction_stage(filtered_docs, config)
# insights_df = run_insight_stage(question, extracted_texts, config)
# forecast = run_forecasting_stage(question, insights_df, config)