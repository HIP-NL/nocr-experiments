# Utrecht 1899 VLM Experiments - Evaluation Script
#
# This script evaluates the performance of different Gemini models on extracting
# structured data from historical handwritten tax records.
#
# Metrics:
# - Character Error Rate (CER)
# - Cell Error Rate
# - Cost analysis

setwd("~/repos/nocr-experiments")

library("data.table")
library("jsonlite")
library("stringdist")
library("tinyplot")

# Set working directory to script location

# Helper function for nice plots
mypar <- function(...) {
    par(...,
        bty = "l",
        mar = c(4, 3, 2, 1),
        mgp = c(1.7, .5, 0),
        tck = -.01,
        font.main = 1
    )
}

# Model ordering and short names
model_order <- c(
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3-flash-preview"
)

short_models <- c(
    "gemini-2.0-flash-lite"    = "2-fl",
    "gemini-2.0-flash"         = "2-f",
    "gemini-2.5-flash-lite"    = "2.5-fl",
    "gemini-2.5-flash"         = "2.5-f",
    "gemini-3-flash-preview"   = "3-f"
)

# Import predictions
print("Loading predictions...")
pred_files = list.files("./results/predictions", pattern = "\\.json$", full.names = TRUE)
names(pred_files) <- basename(pred_files)

preds <- lapply(pred_files, fromJSON)
preds <- lapply(preds, as.data.table)
preds <- lapply(preds, setnames, "maiden name", "maiden_name", skip_absent = TRUE)
preds <- rbindlist(preds, idcol = "file")

# Import ground truth
print("Loading ground truth...")
gt_files <- list.files("./data/ground_truth", pattern = "\\.json$", full.names = TRUE)
names(gt_files) <- basename(gt_files)

gt <- lapply(gt_files, fromJSON)
gt <- rbindlist(gt, idcol = "file")

# Import metadata
print("Loading metadata...")
meta_files <- list.files("./results/metadata", pattern = "\\.json$", full.names = TRUE)
names(meta_files) <- basename(meta_files)

meta <- lapply(meta_files, fromJSON)
meta <- rbindlist(meta, idcol = "file", fill = TRUE)

# Parse prediction filenames (new format: {image}__{model}__{strategy}__{thinking}.json)
print("Parsing prediction metadata...")
preds[, image := stringi::stri_extract_first_regex(file, "NL-UtHUA_[A-Z0-9_]+")]
preds[, model := stringi::stri_extract_first_regex(file, "gemini-[^_]+")]
preds[, strategy := stringi::stri_extract_first_regex(file, "(zeroshot|fewshot)")]
preds[, thinking := stringi::stri_extract_first_regex(file, "thinking\\d+")]

# Add row numbers for proper alignment
preds[, row := 1:.N, by = list(model, strategy, thinking, image)]

# Parse metadata filenames
print("Parsing metadata...")
meta[, image := stringi::stri_extract_first_regex(file, "NL-UtHUA_[A-Z0-9_]+")]
meta[, model := stringi::stri_extract_first_regex(file, "gemini-[^_]+")]
meta[, strategy := stringi::stri_extract_first_regex(file, "(zeroshot|fewshot)")]
meta[, thinking := stringi::stri_extract_first_regex(file, "thinking\\d+")]

# Parse ground truth filenames
gt[, image := stringi::stri_extract_first_regex(file, "NL-UtHUA_[A-Z0-9_]+")]
gt[, row := 1:.N, by = image]

# Merge predictions with ground truth
print("Merging predictions with ground truth...")
eval <- merge(preds, gt, by = c("image", "row"), suffixes = c("_pred", "_gt"), all.x = TRUE)

# Field pattern for evaluation
field_pattern <- "(volgnummer|title|initials|surname|maiden_name|street|house_number|class|tax)"

# Standardize fields: trim whitespace, normalize initials
print("Standardizing fields...")
eval <- eval[, lapply(.SD, trimws),
    .SDcols = patterns(field_pattern),
    by = list(image, model, strategy, thinking, row)
]
eval[, initials_gt := gsub(" ", "", initials_gt)]
eval[, initials_pred := gsub(" ", "", initials_pred)]

# Convert NA to empty string for comparison
eval[is.na(eval)] <- ""

# Calculate character-level errors using Levenshtein distance
print("Calculating error metrics...")
eval[, char_errors := stringdist(volgnummer_pred, volgnummer_gt) +
    stringdist(title_pred, title_gt) +
    stringdist(initials_pred, initials_gt) +
    stringdist(surname_pred, surname_gt) +
    stringdist(maiden_name_pred, maiden_name_gt) +
    stringdist(street_pred, street_gt) +
    stringdist(house_number_pred, house_number_gt) +
    stringdist(class_pred, class_gt) +
    stringdist(tax_pred, tax_gt)]

# Calculate total character count for normalization
eval[, nchar_row := nchar(
    paste0(
        volgnummer_gt, title_gt, initials_gt, surname_gt, maiden_name_gt,
        street_gt, house_number_gt, class_gt, tax_gt
    )
)]

# Calculate cell-level errors (exact match required)
eval[, cell_errors := (volgnummer_pred != volgnummer_gt) +
    (title_pred != title_gt) +
    (initials_pred != initials_gt) +
    (surname_pred != surname_gt) +
    (maiden_name_pred != maiden_name_gt) +
    (street_pred != street_gt) +
    (house_number_pred != house_number_gt) +
    (class_pred != class_gt) +
    (tax_pred != tax_gt)]

# Summarize by model, strategy, and thinking budget
print("Summarizing results...")
smry <- eval[, list(
    cer = mean(char_errors / nchar_row),
    cell_error_rate = mean(cell_errors / 9),
    n_records = .N
), by = list(model, strategy, thinking)]

# Overall model performance
print("\nOverall model performance (averaged across strategies):")
print(smry[, list(cer = mean(cer), cell_error_rate = mean(cell_error_rate)),
    by = model
][order(cer)])

# Top 5 configurations
print("\nTop 5 configurations:")
print(smry[order(cer)][1:5, list(model, strategy, thinking, cer, cell_error_rate)])

# Reshape for plotting
smry_long <- melt(
    smry,
    measure.vars = c("cer", "cell_error_rate"),
    variable.name = "error_type",
    value.name = "error_rate"
)

smry_long[, model_order := match(model, model_order)]
smry_long <- smry_long[order(model_order)]
smry_long[, short_model := short_models[model]]

# Plot: Error rates by strategy and thinking
print("\nGenerating plot: error rates by strategy...")
pdf(file = "./evaluation/figures/error_rates_by_strategy.pdf", height = 6, width = 10)
mypar()
plt(error_rate ~ model_order | strategy + thinking,
    facet = ~error_type,
    data = smry_long,
    legend = "top!",
    type = "b", pch = 20,
    xaxl = function(x) short_models,
    xaxb = 1:5,
    ylab = "Error Rate",
    xlab = "Model",
    main = "Error Rates by Model, Strategy, and Thinking Budget"
)
dev.off()

# Add cost calculations
print("Calculating costs...")
meta[is.na(thoughts_token_count), thoughts_token_count := 0]
meta[, output_tokens := candidates_token_count + thoughts_token_count]
meta[, input_tokens := prompt_token_count]

# Cost per million tokens (as of pricing date)
meta[
    ,
    cost := fcase(
        model == "gemini-3-flash-preview", output_tokens * 3.0 + input_tokens * 0.50,
        model == "gemini-2.5-flash", output_tokens * 2.5 + input_tokens * 0.30,
        model == "gemini-2.5-flash-lite", output_tokens * 0.4 + input_tokens * 0.10,
        model == "gemini-2.0-flash", output_tokens * 0.4 + input_tokens * 0.10,
        model == "gemini-2.0-flash-lite", output_tokens * 0.3 + input_tokens * 0.075,
        default = NA
    )
]
meta[, cost := cost / 1e6]

# Summarize costs
meta_sum <- meta[, list(
    total_tokens = sum(total_token_count, na.rm = TRUE),
    thought_tokens = sum(thoughts_token_count, na.rm = TRUE),
    cost = sum(cost)
), by = list(strategy, model, thinking)]

# Merge cost data with error metrics
smry_long <- merge(
    smry_long,
    meta_sum,
    by = c("strategy", "model", "thinking")
)

# Cost-accuracy frontier
print("\nCost-accuracy trade-offs:")
print(smry_long[
    error_type == "cer",
    list(cost, model, strategy, thinking, error_rate)
][order(cost)])

setorder(smry_long, model, cost)

# Plot: Error rates by cost
print("\nGenerating plot: error rates by cost...")
pdf(file = "./evaluation/figures/error_rates_by_cost_and_model.pdf", height = 6, width = 10)
mypar()
plt(error_rate ~ cost | model,
    facet = ~error_type,
    data = smry_long,
    legend = "top!",
    type = "b", pch = 20,
    xlab = "Cost (USD)",
    ylab = "Error Rate",
    main = "Error Rates by Cost and Model"
)
dev.off()

cat("\n---------------------------\n")
print("Evaluation complete!")
print("Figures saved to: ./figures/")
cat("\n---------------------------\n")
