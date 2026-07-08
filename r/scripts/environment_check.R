# ============================================================
# BggDeepLearning R environment check
# File: r/scripts/environment_check.R
#
# Usage:
# Rscript r\scripts\environment_check.R
# ============================================================

get_current_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)

  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg[1])
    return(normalizePath(script_path, winslash = "/", mustWork = TRUE))
  }

  return(normalizePath(getwd(), winslash = "/", mustWork = TRUE))
}

find_project_root <- function(start_path) {
  current_path <- start_path

  if (!dir.exists(current_path)) {
    current_path <- dirname(current_path)
  }

  for (i in 1:10) {
    config_file <- file.path(current_path, "configs", "app.yaml")

    if (file.exists(config_file)) {
      return(normalizePath(current_path, winslash = "/", mustWork = TRUE))
    }

    parent_path <- dirname(current_path)

    if (identical(parent_path, current_path)) {
      break
    }

    current_path <- parent_path
  }

  stop("Project root was not found. Please check configs/app.yaml.")
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
}

main <- function() {
  script_path <- get_current_script_path()
  project_root <- find_project_root(script_path)

  log_dir <- file.path(project_root, "outputs", "logs")
  table_dir <- file.path(project_root, "outputs", "tables")
  figure_dir <- file.path(project_root, "outputs", "figures")
  report_dir <- file.path(project_root, "outputs", "reports")

  ensure_dir(log_dir)
  ensure_dir(table_dir)
  ensure_dir(figure_dir)
  ensure_dir(report_dir)

  log_file <- file.path(log_dir, "r_environment_check.txt")

  now_time <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")

  output_lines <- c(
    "============================================================",
    "BggDeepLearning R environment check succeeded",
    "------------------------------------------------------------",
    paste0("Check time: ", now_time),
    paste0("R version: ", R.version.string),
    paste0("R platform: ", R.version$platform),
    paste0("R home: ", R.home()),
    paste0("Project root: ", project_root),
    "------------------------------------------------------------",
    paste0("Log directory: ", log_dir),
    paste0("Table output directory: ", table_dir),
    paste0("Figure output directory: ", figure_dir),
    paste0("Report output directory: ", report_dir),
    "------------------------------------------------------------",
    "Current R module status: basic environment check passed",
    "Next modules: Table 1, group comparison, logistic regression, survival analysis, DCA",
    "============================================================"
  )

  cat(paste(output_lines, collapse = "\n"))
  cat("\n")

  writeLines(output_lines, con = log_file, useBytes = TRUE)
}

main()