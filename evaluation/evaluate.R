# Evaluates the performance and cost of gemini and gemma models on extracting # structured
# data from historical handwritten tax records.

setwd("~/repos/nocr-experiments")

library("data.table")
library("jsonlite")
library("stringdist")
library("tinyplot")

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
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview"
)

short_models <- c(
    "gemini-2.0-flash-lite"           = "2-fl",
    "gemini-2.0-flash"                = "2-f",
    "gemini-2.5-flash-lite"           = "2.5-fl",
    "gemini-2.5-flash"                = "2.5-f",
    "gemini-3-flash-preview"          = "3-f",
    "gemini-3.1-flash-lite-preview"   = "3.1-fl"
)

# Import predictions
pred_files <- list.files("./results/predictions", pattern = "\\.json$", full.names = TRUE)
names(pred_files) <- basename(pred_files)

preds <- lapply(pred_files, fromJSON)
nested = sapply(preds, length) == 1
preds[nested] = lapply(preds[nested], \(x) x[[1]])
preds <- lapply(preds, as.data.table)
preds <- lapply(preds, setnames, "maiden name", "maiden_name", skip_absent = TRUE)
preds <- lapply(preds, setnames, "initial", "initials", skip_absent = TRUE)
preds <- lapply(preds, setnames, "initial_name", "initials", skip_absent = TRUE)
# so you can't really read in the new ones because some of them are
preds <- rbindlist(preds, idcol = "file", fill = TRUE)

# Import predictions along format
preds_format = fread("./results/predictions_format/processed_predictions_format.csv")

# Import ground truth
gt_files <- list.files("./data/ground_truth", pattern = "\\.json$", full.names = TRUE)
names(gt_files) <- basename(gt_files)

gt <- lapply(gt_files, fromJSON)
gt <- rbindlist(gt, idcol = "file")

# Import metadata
meta_files <- list.files("./results/metadata", pattern = "\\.json$", full.names = TRUE)
names(meta_files) <- basename(meta_files)

meta <- lapply(meta_files, fromJSON)
meta <- rbindlist(meta, idcol = "file", fill = TRUE)

names(preds) == names(preds_format)
preds = rbind(preds, preds_format)


img_regex = "NL-UtHUA_[A-Z0-9_]+_[lr]"
mdl_regex = "gem(ini|ma)-[^_]+"

# Parse prediction filenames (new format: {image}__{model}__{strategy}__{thinking}.json)
preds[, image := stringi::stri_extract_first_regex(file, img_regex)]
preds[, model := stringi::stri_extract_first_regex(file, mdl_regex)]
preds[, strategy := stringi::stri_extract_first_regex(file, "(zeroshot|fewshot)")]
preds[, thinking := stringi::stri_extract_first_regex(file, "thinking\\d+")]
preds[, format := stringi::stri_extract_first_regex(file, "(yaml|md)")]
preds[is.na(format), format := "json"]

preds[, .N, by = model]

# Add row numbers for proper alignment
preds[, row := 1:.N, by = list(model, strategy, thinking, image, format)]

# Parse metadata filenames
meta[, image := stringi::stri_extract_first_regex(file, img_regex)]
meta[, model := stringi::stri_extract_first_regex(file, mdl_regex)]
meta[, strategy := stringi::stri_extract_first_regex(file, "(zeroshot|fewshot)")]
meta[, thinking := stringi::stri_extract_first_regex(file, "thinking\\d+")]

# extra formats yaml/md
meta[, format := stringi::stri_extract_first_regex(file, "(yaml|md)")]
meta[is.na(format), format := "json"]

# Add cost calculations, price per scan, input/output prices
meta[is.na(thoughts_token_count), thoughts_token_count := 0]
meta[, output_tokens := candidates_token_count + thoughts_token_count]
meta[, input_tokens := prompt_token_count]

meta[, cost := fcase(
        model == "gemma-4-26b-a4b-it",            output_tokens * 0.33 + input_tokens * 0.06, # or free
        model == "gemma-4-31b-it",                output_tokens * 0.35 + input_tokens * 0.12, # or free
        model == "gemini-3.1-flash-lite-preview", output_tokens * 1.50 + input_tokens * 0.25,
        model == "gemini-3-flash-preview",        output_tokens * 3.00 + input_tokens * 0.50,
        model == "gemini-2.5-flash",              output_tokens * 2.50 + input_tokens * 0.30,
        model == "gemini-2.5-flash-lite",         output_tokens * 0.40 + input_tokens * 0.10,
        model == "gemini-2.0-flash",              output_tokens * 0.40 + input_tokens * 0.10,
        model == "gemini-2.0-flash-lite",         output_tokens * 0.30 + input_tokens * 0.075,
        default = NA
    )
]
meta[, cost := cost / 1e6]

# Summarize costs
meta_sum <- meta[,
    list(
        candidate_tokens = sum(candidates_token_count, na.rm = TRUE),
        total_tokens = sum(total_token_count, na.rm = TRUE),
        thought_tokens = sum(thoughts_token_count, na.rm = TRUE),
        cost = sum(cost)
    ), by = list(strategy, model, thinking, format)]


meta_sum_format = meta_sum
meta_sum = meta_sum[format == "json"]

# Parse ground truth filenames
gt[, image := stringi::stri_extract_first_regex(file, img_regex)]
gt[, row := 1:.N, by = image]

# Merge predictions with ground truth
eval <- merge(preds, gt, by = c("image", "row"), suffixes = c("_pred", "_gt"), all.x = TRUE)

# Field pattern for evaluation
field_pattern <- "(volgnummer|title|initials|surname|maiden_name|street|house_number|class|tax)"

# Standardize fields: trim whitespace, normalize initials
eval <- eval[, lapply(.SD, trimws),
    .SDcols = patterns(field_pattern),
    by = list(image, model, strategy, thinking, format, row)
]
eval[, initials_gt := gsub(" ", "", initials_gt)]
eval[, initials_pred := gsub(" ", "", initials_pred)]

# Convert NA to empty string for comparison
eval[is.na(eval)] <- ""

# Calculate character-level errors using Levenshtein distance
eval[, char_errors :=
    stringdist(volgnummer_pred, volgnummer_gt) +
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
eval[, cell_errors :=
    (volgnummer_pred != volgnummer_gt) +
    (title_pred != title_gt) +
    (initials_pred != initials_gt) +
    (surname_pred != surname_gt) +
    (maiden_name_pred != maiden_name_gt) +
    (street_pred != street_gt) +
    (house_number_pred != house_number_gt) +
    (class_pred != class_gt) +
    (tax_pred != tax_gt)]

# Summarize by model, strategy, thinking budget, and format
smry <- eval[,
    list(
        cer = mean(char_errors / nchar_row),
        cell_error_rate = mean(cell_errors / 9),
        n_records = .N),
    by = list(model, strategy, thinking, format)]

smry <- eval[,
    list(
        cer = sum(char_errors) / sum(nchar_row),
        cell_error_rate = sum(cell_errors / 9) / .N,
        n_records = .N),
    by = list(model, strategy, thinking, format)]

# split summaries
smry_format = smry[format != "json"]
smry_open = smry[grepl("gemma", model)]
smry = smry[format == "json" & !grepl("gemma", model)]

# add other zeroshot back to format for comparison to json
smry_format = rbind(
    smry_format,
    smry[model %in% smry_format$model & strategy == "zeroshot"]
)

# Overall model performance
print("\nOverall model performance (averaged across strategies):")
out = smry[, list(cer = mean(cer), cell_error_rate = mean(cell_error_rate)), by = model]
knitr::kable(out[order(cer)], digits = 3)
writeLines(
    knitr::kable(out[order(cer)], format = "latex", digits = 3),
    con = "./evaluation/tables/performance_by_strategy.tex"
)


# Top 5 configurations
print("\nTop 5 configurations:")
knitr::kable(smry[order(cer)][1:5, list(model, strategy, thinking, cer, cell_error_rate)], digits = 3)
writeLines(
    knitr::kable(
        smry[order(cer)][1:5, list(model, strategy, thinking, cer, cell_error_rate)],
        format = "latex",
        caption = "Top 5 models and strategies by error rate.",
        label = "top5cer",
        digits = 3
    ),
    con = "./evaluation/tables/top5_cer.tex"
)


# Reshape for plotting
smry_long <- melt(
    smry,
    measure.vars = c("cer", "cell_error_rate"),
    variable.name = "error_type",
    value.name = "error_rate"
)


smry_long[, model_order := match(model, model_order)]
smry_long = smry_long[order(model_order)]
smry_long[, short_model := short_models[model]]

# Plot: Error rates by strategy and thinking
pdf(file = "./evaluation/figures/error_rates_by_strategy.pdf", height = 6, width = 10)
mypar()
plt(error_rate ~ model_order | strategy + thinking,
    facet = ~error_type,
    data = smry_long,
    legend = "top!",
    type = "b", pch = 20,
    xaxl = function(x) short_models,
    xaxb = 1:length(short_models),
    ylab = "Error Rate",
    xlab = "Model",
    main = NA
)
dev.off()


# Merge cost data with error metrics
smry_long <- merge(
    smry_long,
    meta_sum,
    by = c("strategy", "model", "thinking")
)

# Cost-accuracy frontier
out = smry_long[error_type == "cer", list(model, strategy, thinking, cost, error_rate)][order(cost)]
knitr::kable(out)
writeLines(
    knitr::kable(
        out[error_rate < 0.1],
        format = "latex",
        caption = "Model cost and performance for models with with at least ten percent CER, ordered by costs",
        label = "costs",
        digits = 3
    ),
    con = "./evaluation/tables/cost_accuracy_tradeoffs.tex"
)


setorder(smry_long, model, cost)
toplot = smry_long[error_type == "cer"]
toplot[, strategy2 := 1:.N, by = list(model)]

# Plot: Error rates by cost
pdf(file = "./evaluation/figures/error_rates_by_cost_and_model.pdf", height = 6, width = 7)
mypar()
plt(error_rate ~ cost | short_model,
    # facet = ~ error_type,
    data = toplot,
    legend = "top!",
    type = "b",
    pch = 20,
    xlab = "Cost (USD)",
    ylab = "Error Rate",
    main = NA
)
dev.off()

# Pareto frontier: best achievable CER at or below each cost threshold
toplot_pareto <- smry_long[error_type == "cer"]
setorder(toplot_pareto, cost)
toplot_pareto[, running_min_cer := cummin(error_rate)]

pdf(file = "./evaluation/figures/error_rates_by_cost_and_model_pareto.pdf", height = 6, width = 7)
mypar()
plt(error_rate ~ cost | short_model,
    data = toplot,
    legend = "top!",
    type = "p",
    pch = 20,
    xlab = "Cost (USD)",
    ylab = "Error Rate (CER)",
    main = NA
)
plt_add(running_min_cer ~ cost, data = toplot_pareto, type = "s")
dev.off()

pdf(file = "./evaluation/figures/error_rates_by_cost_and_strategyl.pdf", height = 6, width = 10)
mypar()
plt(error_rate ~ cost | strategy,
    facet = ~ error_type,
    data = smry_long,
    legend = "top!",
    type = "p", pch = 20,
    xlab = "Cost (USD)",
    ylab = "Error Rate",
    main = NA
)
dev.off()

# === accuracy by format ===
toplot = smry_format[order(format, -model, -thinking)]
toplot[, model_order := 1:.N, by = format]
toplot[, model_thinking := paste0(model, "\n", thinking)]
model_thinking = unique(toplot$model_thinking)

pdf(file = "./evaluation/figures/cer_by_format.pdf", height = 7, width = 7)
plt(model_order ~ cer | format, data = toplot[!is.na(cer)],
    type = "b", pch = 20,
    yaxl = function(x) model_thinking,
    yaxb = 1:length(model_thinking),
    xlab = "Error Rate",
    ylab = "",
    legend = "bottomright",
    theme = tinytheme(las = 1, mar = c(4, 13, 2, 1), bty = "l")
)
dev.off()

out = meta[model %in% preds[format == "md", model] & strategy == "zeroshot",
    list(input = mean(prompt_token_count), output = mean(candidates_token_count)), by = format]
out[, output_as_share_max := output / max(output)]

knitr::kable(out, digits = 2)

# === Open models by size ===

smry_open = smry_open[order(strategy, -model, -thinking)]
smry_open[, parameters := stringi::stri_extract_first_regex(model, "\\d+[Bb]-")]
smry_open[, parameters := as.integer(stringi::stri_extract_first_regex(parameters, "\\d+"))]
setorder(smry_open, parameters)

pdf(file = "./evaluation/figures/cer_by_size_open.pdf", height = 6, width = 7)
plt(cer ~ parameters | strategy, data = smry_open,
    log = "x",
    type = "b", pch = 20,
    xlab = "Parameters",
    ylab = "CER",
    legend = "topright",
    theme = tinytheme(las = 1, mar = c(4, 13, 2, 1), bty = "l")
)
dev.off()

cat("\n---------------------------\n")
print("Done!")
cat("\n---------------------------\n")
