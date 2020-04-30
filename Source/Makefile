.SECONDARY:

SHELL = /bin/sh
CC    = gcc

# Source Files
SRCDIR := src
SRCEXT := c
SRCS := $(wildcard $(SRCDIR)/*.$(SRCEXT))

# Sources for the helper library functions
HLPRDIR := $(SRCDIR)/helper
SRCSHELPER := $(wildcard $(HLPRDIR)/*.$(SRCEXT))

# Target helper library
LIBHELPER := libhelperfuncs.a

# Directory names
DBGDIR := debug
RELDIR := release
OUTDIR := out
INCDIR := inc
LIBDIR := lib
DBGLIBDIR := $(DBGDIR)/$(LIBDIR)
RELLIBDIR := $(RELDIR)/$(LIBDIR)

# Objects for the helper library functions
OBJDIR := obj
OBJEXT := o
OBJSHELPER := $(SRCSHELPER:$(HLPRDIR)/%.$(SRCEXT)=%.$(OBJEXT))
DBGOBJSHELPER := $(addprefix $(DBGDIR)/$(OBJDIR)/,$(OBJSHELPER))
RELOBJSHELPER := $(addprefix $(RELDIR)/$(OBJDIR)/,$(OBJSHELPER))

# Executables
BIN := $(SRCS:$(SRCDIR)/%.$(SRCEXT)=%)


# Flags
CFLAGS := -pthread -march=native -Wno-unused-result
DBGFLAGS := -O0 -g
RELFLAGS := -Ofast -funroll-loops
INCPATH := -I$(INCDIR)
DBGLIBPATH := -L$(DBGDIR)/$(LIBDIR)
RELLIBPATH := -L$(RELDIR)/$(LIBDIR)
LIBS := -lm
LIBSUSR := -lm -lhelperfuncs

# Default target
.PHONY: all
all: release


# Debug 
.PHONY: debug
debug: $(addprefix $(DBGDIR)/,$(BIN)) | $(OUTDIR)/.

$(DBGDIR)/%: $(SRCDIR)/%.$(SRCEXT) $(DBGLIBDIR)/$(LIBHELPER) | $(DBGDIR)/.
	$(CC) $(CFLAGS) $(DBGFLAGS) $(INCPATH) $(DBGLIBPATH) -o $@ $< $(LIBSUSR)

# Release
.PHONY: release
release: $(addprefix $(RELDIR)/,$(BIN)) | $(OUTDIR)/.

$(RELDIR)/%: $(SRCDIR)/%.$(SRCEXT) $(RELLIBDIR)/$(LIBHELPER) | $(RELDIR)/.
	$(CC) $(CFLAGS) $(RELFLAGS) $(INCPATH) $(RELLIBPATH) -o $@ $< $(LIBSUSR)


# Debug helper library
$(DBGLIBDIR)/$(LIBHELPER): $(DBGOBJSHELPER) | $(DBGDIR)/$(LIBDIR)/.
	ar -rcs $(DBGDIR)/$(LIBDIR)/$(LIBHELPER) $(DBGOBJSHELPER)

# Release helper library
$(RELLIBDIR)/$(LIBHELPER): $(RELOBJSHELPER) | $(RELDIR)/$(LIBDIR)/.
	ar -rcs $(RELDIR)/$(LIBDIR)/$(LIBHELPER) $(RELOBJSHELPER)


# Debug helper function objects
$(DBGDIR)/$(OBJDIR)/%.$(OBJEXT): $(HLPRDIR)/%.$(SRCEXT) | $(DBGDIR)/$(OBJDIR)/.
	$(CC) $(CFLAGS) $(DBGFLAGS) $(INCPATH) -c -o $@ $< $(LIBS)

# Release helper function objects
$(RELDIR)/$(OBJDIR)/%.$(OBJEXT): $(HLPRDIR)/%.$(SRCEXT) | $(RELDIR)/$(OBJDIR)/.
	$(CC) $(CFLAGS) $(RELFLAGS) $(INCPATH) -c -o $@ $< $(LIBS)


# Make directories
%/.:
	@mkdir -p $*


# Clean all
.PHONY: clean
clean:
	rm -rf $(RELDIR) $(DBGDIR)

