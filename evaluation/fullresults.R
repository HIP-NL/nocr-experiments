rm(list = ls())

library(readxl)
library(jsonlite)
library(data.table)
library(stringdist)
library(stringi)
library(knitr)
library(tinyplot) # Using tinyplot for base R-like plotting as requested

# -------------------------------------------------------------------------
# 0. Setup Output Directories
# -------------------------------------------------------------------------

print(getwd())

dir.create("evaluation/tables", showWarnings = FALSE, recursive = TRUE)
dir.create("evaluation/figures", showWarnings = FALSE, recursive = TRUE)

# -------------------------------------------------------------------------
# 1. Helper Functions
# -------------------------------------------------------------------------
mypar <- function(...) {
    par(...,
        bty = "l",
        mar = c(4, 3, 2, 1),
        mgp = c(1.7, .5, 0),
        tck = -.01,
        font.main = 1
    )
}

#' Calculate transformations between two strings using adist
get_trafos = function(x, y) {
    if (is.na(x) || is.na(y)) {
        return(NA_character_)
    }
    dst = adist(x, y, counts = TRUE)
    return(attr(dst, "trafos"))
}

#' Extract maximum consecutive operations (e.g., Deletions or Insertions)
get_max_consecutive = function(trafo_str, op_char) {
    if (is.na(trafo_str)) {
        return(NA_integer_)
    }
    regex_pattern = paste0(op_char, "+")
    matches = stri_extract_all_regex(trafo_str, regex_pattern)[[1]]
    if (any(is.na(matches))) {
        return(0L)
    }
    return(max(nchar(matches), na.rm = TRUE))
}

#' Get top most frequent mistakes for a given column
get_top_mistakes = function(dt, col_name, top_n = 10) {
    orig_col = paste0(col_name, "_orig")
    corr_col = paste0(col_name, "_corr")

    mistakes = dt[get(orig_col) != get(corr_col),
        .(Count = .N),
        by = c(orig_col, corr_col)
    ]

    setnames(mistakes, c(orig_col, corr_col),  c("orig_col", "corr_col"))

    mistakes = mistakes[order(-Count)]
    return(head(mistakes, top_n))
}

# -------------------------------------------------------------------------
# 2. Data Loading and Preprocessing
# -------------------------------------------------------------------------

cat("\nLoading and preprocessing data...\n")

path_corrected = "~/data/hipnl/utrecht/utrecht/1899_nocr/utrecht_1899_merged.xlsx"
sheets = excel_sheets(path_corrected)
sheets = sheets[-c(1:2)] # Exclude overview sheets if any
xllist = lapply(sheets, read_xlsx, path = path_corrected)
dc = rbindlist(xllist, fill = TRUE)

# Rename columns in Corrected Data to match original JSON Data before merging
dc[, street := NULL]
setnames(dc,
    old = c("street_original", "house_nr_street"),
    new = c("street", "house_number"),
    skip_absent = TRUE
)

path_json = "~/repos/nocr/utrecht/1899/response"
files = list.files(path = path_json, pattern = "*.json", recursive = TRUE, full.names = TRUE)
files = sort(files)
names(files) = gsub("response_|\\.json", "", files)

datlist = lapply(files, fromJSON)
datlist = lapply(datlist, as.data.table)
do = rbindlist(datlist, fill = TRUE, idcol = "half_page_file")

do[, inv_no := stri_extract_first_regex(half_page_file, "6\\d{3}")]
do[, json := paste0(stri_extract_first_regex(half_page_file, "NL.*"), ".json")]

# Row counter for each page for merging
dc[, i := 1:.N, by = .(inv_no, json)]
do[, i := 1:.N, by = .(inv_no, json)]

# -------------------------------------------------------------------------
# 3. Data Merging
# -------------------------------------------------------------------------

cat("Merging original predictions and ground truth corrections...\n")

d = merge(
    do,
    dc,
    by = c("json", "inv_no", "i"),
    suffixes = c("_orig", "_corr")
)

eval_cols = c("volgnummer", "title", "initials", "surname", "street", "house_number", "class", "tax")

# Ensure all relevant columns are character and handle NAs
for (col in eval_cols) {
    orig_col = paste0(col, "_orig")
    corr_col = paste0(col, "_corr")

    if (orig_col %in% names(d)) d[, (orig_col) := as.character(get(orig_col))]
    if (corr_col %in% names(d)) d[, (corr_col) := as.character(get(corr_col))]

    d[is.na(get(orig_col)), (orig_col) := ""]
    d[is.na(get(corr_col)), (corr_col) := ""]
}

# -------------------------------------------------------------------------
# 4. Overall Error Rates (CER / Cell Error Rate)
# -------------------------------------------------------------------------

cat("\n--- Overall Performance ---\n")

d[, fullstring_orig := paste0(volgnummer_orig, title_orig, initials_orig, surname_orig, street_orig, house_number_orig, class_orig, tax_orig)]
d[, fullstring_corr := paste0(volgnummer_corr, title_corr, initials_corr, surname_corr, street_corr, house_number_corr, class_corr, tax_corr)]
d[, nchar_fullstring_orig := nchar(fullstring_orig)]

d[nchar_fullstring_orig == 0, nchar_fullstring_orig := 1]
d[, overall_cer := stringsim(fullstring_orig, fullstring_corr, method = "lv")]

mean_cer = round(mean(1 - d$overall_cer, na.rm = TRUE) * 100, 2)
median_cer = round(median(1 - d$overall_cer, na.rm = TRUE) * 100, 2)

cat(sprintf("Mean Character Error Rate (CER): %s%%\n", mean_cer))
cat(sprintf("Median CER: %s%%\n", median_cer))

d[, fullstring_nostreet_orig := paste0(volgnummer_orig, title_orig, initials_orig, surname_orig, house_number_orig, class_orig, tax_orig)]
d[, fullstring_nostreet_corr := paste0(volgnummer_corr, title_corr, initials_corr, surname_corr, house_number_corr, class_corr, tax_corr)]
d[, nchar_fullstring_nostreet_orig := nchar(fullstring_nostreet_orig)]

d[nchar_fullstring_nostreet_orig == 0, nchar_fullstring_nostreet_orig := 1]
d[, overall_cer_nostreet := stringsim(fullstring_nostreet_orig, fullstring_nostreet_corr, method = "lv")]

mean_cer_nostreet = round(mean(1 - d$overall_cer_nostreet, na.rm = TRUE) * 100, 2)
median_cer_nostreet = round(median(1 - d$overall_cer_nostreet, na.rm = TRUE) * 100, 2)

cat(sprintf("Mean Character Error Rate (CER) omitting street: %s%%\n", mean_cer_nostreet))
cat(sprintf("Median CER omitting street: %s%%\n", median_cer_nostreet))

total_cells = nrow(d) * length(eval_cols)
incorrect_cells = 0
for (col in eval_cols) {
    incorrect_cells = incorrect_cells + sum(d[[paste0(col, "_orig")]] != d[[paste0(col, "_corr")]], na.rm = TRUE)
}
overall_cell_error = round((incorrect_cells / total_cells) * 100, 2)
cat(sprintf("Overall Cell Error Rate: %s%%\n", overall_cell_error))

# -------------------------------------------------------------------------
# 5. Field-Level Distance, Similarity & Structural Mistakes
# -------------------------------------------------------------------------

cat("\n--- Field-Level Analysis ---\n")

error_types_list = list()

for (col in eval_cols) {
    orig_col = paste0(col, "_orig")
    corr_col = paste0(col, "_corr")
    dist_col = paste0(col, "_dist")
    sim_col = paste0(col, "_sim")
    trafos_col = paste0(col, "_trafos")

    # Distance and similarity
    d[, (dist_col) := stringdist(get(orig_col), get(corr_col), method = "lv")]
    d[, (sim_col) := stringsim(get(orig_col), get(corr_col), method = "lv")]

    # Structural transformations
    trafos_res = mapply(get_trafos, d[[orig_col]], d[[corr_col]], SIMPLIFY = FALSE)
    d[, (trafos_col) := sapply(trafos_res, function(x) if (length(x) > 0) x[1] else NA_character_)]

    d[, paste0(col, "_max_D") := sapply(get(trafos_col), get_max_consecutive, op_char = "D")]
    d[, paste0(col, "_max_I") := sapply(get(trafos_col), get_max_consecutive, op_char = "I")]
    d[, paste0(col, "_max_S") := sapply(get(trafos_col), get_max_consecutive, op_char = "S")]

    # Categorize error types based on distances
    d[, paste0(col, "_dist_error_type") := fcase(
        get(dist_col) == 0, "None",
        get(sim_col) >= 0.7 & get(dist_col) > 0, "Transcription",
        get(sim_col) < 0.7, "Structural/Major",
        default = "Unknown"
    )]

    # Categorize based on structural transformations (Trafos trick)
    d[, paste0(col, "_trafos_error_type") := fcase(
        get(dist_col) == 0, "None",
        get(paste0(col, "_max_D")) > 4 & get(paste0(col, "_max_I")) > 4, "Omission and inclusion",
        get(paste0(col, "_max_D")) > 4, "Inclusion",
        get(paste0(col, "_max_I")) > 4, "Omission",
        default = "Transcription"
    )]

    # Collect for tables
    err_trafo = d[, .N, by = c(paste0(col, "_trafos_error_type"))]
    setnames(err_trafo, 1, "Error_Type")
    err_trafo[, Column := col]

    error_types_list[[col]] = err_trafo
}

# 5.1 Export Similarity Means and Cell Error Rates Table
sim_cols = grep("_sim$", names(d), value = TRUE)
sim_means = d[, lapply(.SD, mean, na.rm = TRUE), .SDcols = sim_cols]

# Reshape for presentation
sim_means_melted = melt(sim_means, measure.vars = sim_cols, variable.name = "Column", value.name = "Mean_Similarity")
sim_means_melted[, Column := gsub("_sim", "", Column)]

# Calculate Cell Error Rates per column
cer_list = list()
for (col in eval_cols) {
    orig_col = paste0(col, "_orig")
    corr_col = paste0(col, "_corr")
    incorrect = sum(d[[orig_col]] != d[[corr_col]], na.rm = TRUE)
    cer_list[[col]] = incorrect / nrow(d)
}
cer_dt = data.table(Column = names(cer_list), Cell_Error_Rate = unlist(cer_list))

# Merge metrics
sim_means_melted = merge(sim_means_melted, cer_dt, by = "Column")

writeLines(
    kable(
        sim_means_melted,
        format = "latex",
        digits = 3,
        booktabs = TRUE,
        caption = "Mean Similarity Scores and Cell Error Rates per Column",
        label = "fullperf",
        position = "!ht"
    ),
    "evaluation/tables/column_performance_metrics.tex"
)
cat("Exported table: column_performance_metrics.tex\n")

# 5.2 Export Structural Error Types Table
trafos_combined = rbindlist(error_types_list)
trafos_wide = dcast(trafos_combined, Column ~ Error_Type, value.var = "N", fill = 0)

setcolorder(trafos_wide, c("Column", "None", "Transcription", "Omission", "Inclusion", "Omission and inclusion"))

writeLines(
    kable(
        trafos_wide,
        position = "!ht",
        valign = "!ht",
        format = "latex",
        booktabs = TRUE,
        label = "trafos",
        caption = "Distribution of Error Types (Structural Transformations) per Column"
    ),
    "evaluation/tables/error_types_trafos.tex"
)
cat("Exported table: error_types_trafos.tex\n")

# 5.3 Export example carried forward table
locf_example = d[json == "NL-UtHUA_A376075_000013_r.jpg.json", list(street_orig, street_corr, street_sim)]
writeLines(
    kable(locf_example, digits = 3, format = "latex", booktabs = TRUE, caption = "Example of carried forward, original json response and corrected data"),
    "evaluation/tables/locf_example.tex"
)
cat("Exported table: locf_example.tex\n")


# -------------------------------------------------------------------------
# 6. Deep Dive: Common Mistakes in Critical Columns
# -------------------------------------------------------------------------

cat("\n--- Extracting Top Mistakes ---\n")

class_mistakes = get_top_mistakes(d, "class")
tax_mistakes = get_top_mistakes(d, "tax")
volgnummer_mistakes = get_top_mistakes(d, "volgnummer")
surname_mistakes = get_top_mistakes(d, "surname")
initials_mistakes = get_top_mistakes(d, "initials")

# class_mistakes[, Field := "Class"]
# tax_mistakes[, Field := "Tax"]

top_mistakes = rbindlist(
    list(
        Class = class_mistakes[1:4],
        Tax = tax_mistakes[1:4],
        Initials = initials_mistakes[1:4],
        Volgnummer = volgnummer_mistakes[1:4],
        Surname = surname_mistakes[1:4]
    ),
    idcol = "Field",
    fill = TRUE)
setnames(top_mistakes, c("orig_col", "corr_col"), c("Original", "Corrected"), skip_absent = TRUE)
setorder(top_mistakes, -Count)

writeLines(
    kable(
        top_mistakes[1:10],
        format = "latex",
        booktabs = TRUE,
        label = "mistakes",
        caption = "The 10 most frequent mistakes in the automated transcription of the 1899 Utrecht tax records"
    ),
    "evaluation/tables/top_mistakes_class_tax.tex"
)
cat("Exported table: top_mistakes_class_tax.tex\n")

# -------------------------------------------------------------------------
# 7. Generate Figures (PDFs) via tinyplot
# -------------------------------------------------------------------------

cat("\nGenerating figures...\n")

# Histogram for Surname Similarity
pdf("evaluation/figures/surname_street_sim_hist.pdf", width = 8, height = 5)
mypar(mfrow = c(1, 2))
plt(~surname_sim,
    data = d,
    type = "histogram",
    breaks = seq(0, 1, by = 0.05),
    main = "Similarity Scores for 'Surname' Column",
    xlab = "Similarity Score (LV)",
    ylab = "Frequency"
)
plt(~ street_sim,
    data = d,
    type = "histogram",
    breaks = seq(0, 1, by = 0.05),
    main = "Similarity Scores for 'Street' Column",
    xlab = "Similarity Score (LV)",
    ylab = "Frequency"
)
dev.off()

cat("Exported figures: surname_sim_hist.pdf and street_sim_hist.pdf\n")
cat("\nPipeline complete!\n")
