rm(list = ls())

library(readxl)
library(jsonlite)
library(data.table)
library(stringdist)
library(stringi)

# -------------------------------------------------------------------------
# 1. Helper Functions
# -------------------------------------------------------------------------

get_trafos = function(x, y) {
    if (is.na(x) || is.na(y)) {
        return(NA_character_)
    }
    dst = adist(x, y, counts = TRUE)
    return(attr(dst, "trafos"))
}

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

# -------------------------------------------------------------------------
# 2. Data Loading and Preprocessing
# -------------------------------------------------------------------------

path_corrected = "~/data/hipnl/utrecht/utrecht/1899_nocr/utrecht_1899_merged.xlsx"
sheets = excel_sheets(path_corrected)
sheets = sheets[-c(1:2)]
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

# Row counter for each page for merg
dc[, i := 1:.N, by = .(inv_no, json)]
do[, i := 1:.N, by = .(inv_no, json)]

# -------------------------------------------------------------------------
# 3. Data Merging
# -------------------------------------------------------------------------

d = merge(
    do,
    dc,
    by = c("json", "inv_no", "i"),
    suffixes = c("_orig", "_corr")
)

# To check: not all these were actually corrected
# From my mail: initials, surname, street, house_numer, tax
eval_cols = c("volgnummer", "title", "initials", "surname", "street", "house_number", "class", "tax")

# Ensure all relevant columns are character
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

d[, fullstring_orig := paste0(volgnummer_orig, title_orig, initials_orig, surname_orig, street_orig, house_number_orig, class_orig, tax_orig)]
d[, fullstring_corr := paste0(volgnummer_corr, title_corr, initials_corr, surname_corr, street_corr, house_number_corr, class_corr, tax_corr)]
d[, nchar_fullstring_orig := nchar(fullstring_orig)]

d[nchar_fullstring_orig == 0, nchar_fullstring_orig := 1]

d[, overall_cer := stringsim(fullstring_orig, fullstring_corr, method = "lv")]

cat("\n--- Overall Performance ---\n")
cat("Mean Character Error Rate (CER):", round(mean(1 - d$overall_cer, na.rm = TRUE) * 100, 2), "%\n")
cat("Median CER:", round(median(1 - d$overall_cer, na.rm = TRUE) * 100, 2), "%\n")

total_cells = nrow(d) * length(eval_cols)
incorrect_cells = 0
for (col in eval_cols) {
    incorrect_cells = incorrect_cells + sum(d[[paste0(col, "_orig")]] != d[[paste0(col, "_corr")]], na.rm = TRUE)
}
cat("Overall Cell Error Rate:", round((incorrect_cells / total_cells) * 100, 2), "%\n")

# -------------------------------------------------------------------------
# 5. Field-Level Distance, Similarity & Structural Mistakes
# -------------------------------------------------------------------------

cat("\n--- Field-Level Analysis ---\n")

for (col in eval_cols) {
    orig_col = paste0(col, "_orig")
    corr_col = paste0(col, "_corr")
    dist_col = paste0(col, "_dist")
    sim_col = paste0(col, "_sim")
    trafos_col = paste0(col, "_trafos")

    d[, (dist_col) := stringdist(get(orig_col), get(corr_col), method = "lv")]
    d[, (sim_col) := stringsim(get(orig_col), get(corr_col), method = "lv")]

    d[, (trafos_col) := mapply(get_trafos, get(orig_col), get(corr_col))]

    d[, paste0(col, "_max_D") := sapply(get(trafos_col), get_max_consecutive, op_char = "D")]
    d[, paste0(col, "_max_I") := sapply(get(trafos_col), get_max_consecutive, op_char = "I")]
    d[, paste0(col, "_max_S") := sapply(get(trafos_col), get_max_consecutive, op_char = "S")]

    d[, paste0(col, "_dist_error_type") := fcase(
        get(dist_col) == 0, "Perfect",
        get(sim_col) >= 0.7 & get(dist_col) > 0, "Transcription Error",
        get(sim_col) < 0.7, "Structural/Major Mistake",
        default = "Unknown"
    )]

    d[, paste0(col, "_trafos_error_type") := fcase(
        get(dist_col) == 0, "Perfect",
        get(paste0(col, "_max_D")) > 4 & get(paste0(col, "_max_I")) > 4, "Mistaken omission and inclusion ",
        get(paste0(col, "_max_D")) > 4, "Mistaken inclusion",
        get(paste0(col, "_max_I")) > 4, "Mistaken omission",
        default = "Transcription error"
    )]

    cat(sprintf("\nColumn: %s\n", col))
    print(d[, .N, by = c(paste0(col, "_dist_error_type"))][order(-N)])
    print(d[, .N, by = c(paste0(col, "_trafos_error_type"))][order(-N)])
}

# the distinction between dist/sim based classification into small/major mistakes is useful. eg in numbers there is almost never a structural mistake, while the dist measures do report it. You see that street is where the major issue is with structural errors.

d[, .SD, .SDcols = patterns("_dist$")]
d[, rowSums(.SD), .SDcols = patterns("_dist$")]
d[, colMeans(.SD), .SDcols = patterns("_sim")]
d[, sapply(.SD, summary), .SDcols = patterns("_sim")]
d[, lapply(.SD, mean), .SDcols = patterns("_sim")]

d[, colMeans(.SD), .SDcols = patterns("_sim")] |> knitr::kable(digits = 2)

d[, mean(volgnummer_sim)]
d[, mean(title_sim)]
d[, mean(initials_sim)]
d[, mean(surname_sim)]
d[, mean(street_sim)]
d[, mean(house_number_sim)]
d[, mean(class_sim)]
d[, mean(tax_sim)]

d[, mean(street_sim), by = inv_no][order(V1)]
d[, mean(street_sim), by = inv_no][order(V1)]
d[order(street_sim), list(inv_no, json, street_orig, street_corr, street_sim)][1:50]

# street sim seems extraordinarily poor. 
hist(d$surname_sim)
hist(d$street_sim)
# but this can basically be the point: all the performance is pretty good, it's just that the streets have this "fill in the rest of the page which is left empty" dynamic, and if this is mistaken, errors propagate massively. Probably if we give the "keep it empy" option, it's much better

d[, .N, by = street_error_type]

# -------------------------------------------------------------------------
# 6. Deep Dive: Common Mistakes in Critical Columns (Class & Tax)
# -------------------------------------------------------------------------

get_top_mistakes = function(dt, col_name, top_n = 10) {
    orig_col = paste0(col_name, "_orig")
    corr_col = paste0(col_name, "_corr")

    mistakes = dt[get(orig_col) != get(corr_col),
        .(Count = .N),
        by = c(orig_col, corr_col)
    ]

    mistakes = mistakes[order(-Count)]
    return(head(mistakes, top_n))
}

# numeric columns
cat("\n--- Top Mistakes: 'class' column ---\n")
class_mistakes = get_top_mistakes(d, "class")
print(class_mistakes)

cat("\n--- Top Mistakes: 'tax' column ---\n")
tax_mistakes = get_top_mistakes(d, "tax")
print(tax_mistakes)

cat("\n--- Top Mistakes: 'volgnummer' column ---\n")
volgnummer_mistakes = get_top_mistakes(d, "volgnummer")
print(volgnummer_mistakes)


cat("\n--- Top Mistakes: 'surname' column ---\n")
surname_mistakes = get_top_mistakes(d, "surname")
print(surname_mistakes)

cat("\n--- Top Mistakes: 'initials' column ---\n")
initials_mistakes = get_top_mistakes(d, "initials")
print(initials_mistakes)

