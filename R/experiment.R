library(abind)
library(beepr)

source('util.R')

meta.filename    <- 'meta.txt'
params.filename  <- 'params.txt'
results.filename <- 'results.df'

params.default <- function() {
	params <- list()

	params$n.replications <- 10 # num of times to repeat each trial
	params$n.attempts     <- 1  # num of times to repeat a trial if it fails

	params$verbose <- 3.1 # verbosity level
						  # 0:   suppress all output
						  # 1:   print progress headers for each trial
						  # 2:   print time remaining estimates
	            # 3:   print status from w/in trial
						  # _.1: print sampled data

	params$write.results    <- TRUE                               # if TRUE, write to disk
	params$result.root.path <- "/Users/kwinner/Work/Data/Results" # where to create result (sub)directory
	params$experiment.name  <- 'latentcounts'                     # an experiment label (half of resultdir)
	params$timestamp.format <- "%Y-%m-%d %H:%M:%OS6"              # format string for timestamping results

	params$shared.data <- 'none' # \in {'experiment', 'replications', or 'none'}
								 # if 'experiment', then data is generated once and shared between all trials and all experiment values
								 # if 'replications', then new data is generated for each experiment value, and shared between all trials
								 # if 'none', then new data is generated for each trial
	
	return(params)
} # params.default

experiment <- function(trial.fun, 
                       data.generation.fun,
                       params = params.default()) {
	results <- data.frame()

	if (params$verbose >= 1)
		cat(sprintf("Beginning experiment %s\n", params$experiment.name))
	
	# prepare an output directory
	if (params$write.results) {
		# check the parent directory
		if (!file.exists(params$result.root.path)) {
			tryCatch({dir.create(params$result.root.path)},
			error = function(e) {
				if (params$verbose > 0)
					warning(sprintf("Unable to find/create result root: %s. \n Results will not be saved.", 
								    params$result.root.path))
				params$write.results <- FALSE
			})
  	}

  	# name the result directory 
		timestamp <- strftime(Sys.time(), params$timestamp.format)
		if (is.null(params$experiment.name))
			result.dir <- timestamp
		else
			result.dir <- paste(timestamp, ' "', params$experiment.name ,'"', sep="")
		params$result.path <- file.path(params$result.root.path, result.dir)

		if (!file.exists(params$result.path)) {
			tryCatch({dir.create(params$result.path)},
			error = function(e) {
				if (params$verbose > 0)
					warning('Unable to find/create result directory. Results will not be saved.')
				params$write.results <- FALSE
			})
		} else
			if (params$verbose > 0)
				warning('Warning: result directory already exists, results may be overwritten.')
	}

	# assert that experiment.var and experiment.values are consistently sized
	# for doing grid search over 2+ values, try:
	#	expand.grid(list(paramA.values, paramB.values, ...))
	# to generate experiment.values
	n.experiment.vars <- length(params$experiment.var)
	stopifnot((n.experiment.vars == 1 && (  is.list(params$experiment.values) ||     is.vector(params$experiment.values))) ||
						(n.experiment.vars > 1  && (is.matrix(params$experiment.values) || is.data.frame(params$experiment.values))
																		&& ncol(params$experiment.values) == n.experiment.vars))

	if (n.experiment.vars == 1)
		n.experiment.values <- length(params$experiment.values)
	else
		n.experiment.values <- nrow(params$experiment.values)

	# generate data (to be shared across all values of experiment.variable)
	if (params$shared.data == 'experiment') {
		# NOTE: experiment-wide data sharing should only be used if the experiment variable does not affect data generation
	    if (any(grepl('gen', params$experiment.var)))
	    	if (params$verbose > 0)
				warning(sprintf("Experiment-wide data sharing is generally inappropriate when the experiment variable (%s) affects data generation.", params$experiment.variable))
		data <- data.generation.fun(params)

		if (params$verbose %% 1 >= 0.1)
			cat(sprintf("y = %s\n", mat2str(data)))
	}

	# for approximating remaining runtime
	start.time <- proc.time()[3]

	# iterate over all combinations of experiment values
	for (i.experiment.value in 1:n.experiment.values) {
		if (n.experiment.vars == 1)
			trial.vals <- params$experiment.values[i.experiment.value]
		else
			trial.vals <- params$experiment.values[i.experiment.value,]
		
		if (params$verbose >= 1)
			cat(sprintf("Trial set #%d of %d, %s = %s\n", 
															i.experiment.value, 
																		n.experiment.values,
																				mat2str(params$experiment.var),
																						 mat2str(trial.vals)))

		# create a new params object, but overwrite/set the values of the experiment variables
		params.trial <- params
		params.trial[params$experiment.var] <- trial.vals

		# generate data (to be shared across all trials)
		if (params$shared.data == 'replications') {
			data <- data.generation.fun(params.trial)
			if (params$verbose %% 1 >= 0.1)
				cat(sprintf("y = %s\n", mat2str(data)))
		}

		# each trial will be independently repeated n.replications times
		for (i.replication in 1:params$n.replications) {
			if (params$verbose >= 1)
				cat(sprintf("  rep #%d of %d...", i.replication, params$n.replications))

			# generate single-use data
			if (params$shared.data == 'none') {
				data <- data.generation.fun(params.trial)
				if (params$verbose %% 1 >= 0.1)
					cat(sprintf("\n    y = %s\n   ", mat2str(data)))
			}

			# in case computation crashes, we have the option to re-attempt a failed trial
			i.attempt <- 1
			while (i.attempt <= params$n.attempts) {
				if(i.attempt > 1 && params$verbose >= 1)
					cat(sprintf("\n    attempt %d...", i.attempt))

				# run the trial (with error handling)
				withCallingHandlers({
					result.trial <- trial.fun(data, params.trial)
					
					# record the number of attempts it took
					result.trial[,'n.attempts'] <- i.attempt

					# record the experiment variables, then reorder the columns
					n.primary.results <- ncol(result.trial)
					result.trial[,params$experiment.var] <- trial.vals
					result.trial <- result.trial[,c((n.primary.results+1):(n.primary.results+n.experiment.vars),
																					1:n.primary.results)]

					# append this trial to the complete record
					results <- rbind(results, result.trial)

					if (params$verbose >= 1)
						cat(sprintf(" Done."))
					break # while (i.attempt <= params$n.attempts)
				}, error = function(e) {
					if (params$verbose >= 1)
						cat(sprintf(" Error: %s.", e))
					next  # while (i.attempt <= params$n.attempts)
				})
			} # while (i.attempt <= params$n.attempts)

			if (params$verbose >= 2) {
				overall.elapsed.time <- proc.time()[3] - start.time
      			percent.completed <- ((((i.experiment.value - 1) * params$n.replications) + i.replication) / 
                		                (n.experiment.values     * params$n.replications))
      			expected.remaining.time <- overall.elapsed.time * (1 - percent.completed) / percent.completed

      			cat(sprintf(' %d%% complete, remaining time ~= %s\n', round(100 * percent.completed), s2hms(expected.remaining.time)))
			} else if(params$verbose %/% 1 == 1) {
				cat(sprintf('\n'))
			}
		} # for (i.replication in 1:params$n.replications)
	} # for (i.experiment.value in 1:n.experiment.values)

	# audio alert (non-critical)
	if(params$verbose > 0)
		tryCatch({beep()}, error=function(e){}, warning=function(w){})

	if (params$write.results) {
	  results.fullpath <- file.path(params$result.path, results.filename)
		write.table(results, file = results.fullpath)
    
		params.fullpath  <- file.path(params$result.path, params.filename)
		write.params(params, params.fullpath)
		
		repo.version <- system("git rev-parse --short HEAD", intern = TRUE)
		meta <- list('repo.version' = repo.version, 
		             'results.fullpath' = results.fullpath, 
		             'params.fullpath'  = params.fullpath)
		write.params(meta, file.path(params$result.path, meta.filename))

		return(params$result.path)
	} else
		return(results)
} # experiment

write.params <- function(params, file.path) {
  file <- file(file.path, 'w')
  
  writeLines(mapply(function(name, val) {
                      if(is.numeric(val) && is.matrix(val))
                        sprintf("%s: %s", name, mat2str(val))
                      else
                        sprintf("%s: %s", name, toString(val))}, 
                    names(params), 
                    params),
             file)
  
  close(file)
} # write.params

load.results <- function(result.dir) {
	return(read.table(file.path(result.dir, results.filename)))
} # load.results

load.params  <- function(result.dir) {
  # read the raw file
	lines <- readLines(file.path(result.dir, params.filename))

	# parse each line
	params <- sapply(lines, function(line) {
	                         # split lines by 'KEY: VALUE'
                	         split <- regmatches(line, regexpr(': ', line), invert = TRUE)[[1]]
                	         
                	         # parse value
                	         if (grepl('\\[.*\\]', split[2])) {
                	           # parse a matrix
                	           setNames(list(str2mat(split[2])), split[1])
                	         } else if (grepl(', ', split[2])) {
                	           # parse a list
                	           split.list <- strsplit(split[2], ', ')[[1]]
                	           
                	           # check if the list is a numeric vector
                	           split.asnum <- suppressWarnings(as.numeric(split.list))
                	           if (all(!is.na(split.asnum)))
                	             split.list <- split.asnum
                	           setNames(list(split.list), split[1])
                           } else if (!is.na(suppressWarnings(as.numeric(split[2])))) {
                             # parse a numeric value
                             setNames(list(as.numeric(split[2])), split[1])
                           } else if (!is.na(suppressWarnings(as.logical(split[2])))) {
                             # parse a logical value
                             setNames(list(as.logical(split[2])), split[1])
                	         } else {
                             # parse (w/o processing) a string
                	           setNames(list(split[2]), split[1])
                           }},
                	       USE.NAMES = FALSE) # names are handled explicitly, don't infer them
	
	return(params)
} # load.params