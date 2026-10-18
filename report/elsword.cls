%% elsword.cls -- LaTeX version of the Elsevier Word article template used for this report.
%% US Letter, 1.41 in side and 1.25 in top/bottom margins, two columns (0.33 in gap), Times,
%% 9 pt text on 10 pt leading, 10 pt bold section and italic subsection headings, 8 pt captions,
%% tables and references, one-column front matter with rules around the abstract.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{elsword}[2026/10/12 Elsevier Word-template layout]
\LoadClass[letterpaper,10pt]{article}

\RequirePackage[letterpaper,top=1.25in,bottom=1.25in,left=1.41in,right=1.41in,
  columnsep=0.33in,footskip=0.45in]{geometry}
\RequirePackage[T1]{fontenc}
\RequirePackage{mathptmx}
\renewcommand\ttdefault{pcr}
\RequirePackage{titlesec}
\RequirePackage{caption}
\RequirePackage{multicol}
\RequirePackage{float}
\setlength\columnsep{0.33in}
\setlength\multicolsep{4pt plus 2pt minus 1pt}
\raggedcolumns
%% Figures and tables run across both columns at the place where they appear in the text: the source
%% closes the two-column block, sets the float with [H] and reopens it, like the continuous section
%% breaks in the Word template.
\RequirePackage[numbers,sort&compress]{natbib}

%% ---- font sizes (Word template: body 9/10 pt, tables, captions and references 8 pt)
\renewcommand\normalsize{\@setfontsize\normalsize{9}{10.5}%
  \abovedisplayskip 6pt plus 2pt minus 2pt\belowdisplayskip\abovedisplayskip
  \abovedisplayshortskip 3pt plus 1pt\belowdisplayshortskip 3pt plus 1pt}
\renewcommand\small{\@setfontsize\small{8.5}{10}}
\renewcommand\footnotesize{\@setfontsize\footnotesize{8}{9.5}}
\renewcommand\scriptsize{\@setfontsize\scriptsize{7}{8.5}}
\renewcommand\tiny{\@setfontsize\tiny{6}{7}}
\normalsize
\setlength\parindent{0.15in}
\setlength\parskip{0pt}
\setlength\footnotesep{8pt}
\setlength\intextsep{8pt plus 2pt minus 2pt}








%% ---- headings: "1. Introduction" (10 pt bold), "1.1. Background" (10 pt italic)
\renewcommand\thesection{\arabic{section}}
\titleformat{\section}{\fontsize{10}{12}\selectfont\bfseries\raggedright}{\thesection.}{0.5em}{}
\titlespacing*{\section}{0pt}{12pt plus 2pt}{5pt}
\titleformat{\subsection}{\fontsize{10}{12}\selectfont\itshape\raggedright}{\thesubsection.}{0.5em}{}
\titlespacing*{\subsection}{0pt}{9pt plus 2pt}{4pt}
\titleformat{\subsubsection}{\fontsize{10}{12}\selectfont\itshape}{\thesubsubsection.}{0.5em}{}
\g@addto@macro\appendix{%
  \titleformat{\section}{\fontsize{10}{12}\selectfont\bfseries\raggedright}{Appendix \thesection.}{0.5em}{}}

%% ---- captions: "Fig. 1. ..." below figures; "Table 1" on its own line above tables
\captionsetup{font=footnotesize,labelfont=rm,justification=justified,singlelinecheck=true,skip=4pt}
\captionsetup[figure]{name=Fig.,labelsep=period}
\captionsetup[table]{labelsep=newline,singlelinecheck=false,justification=raggedright,position=top,skip=3pt}

%% ---- references: numbered [1], 8 pt
\renewcommand\bibfont{\footnotesize\raggedright}
\setlength\bibsep{1pt plus 0.5pt}
\renewcommand\bibsection{\section*{References}}

%% ---- front matter
\newcommand\@affiliation{}\newcommand\affiliation[1]{\renewcommand\@affiliation{#1}}
\newcommand\@email{}\newcommand\email[1]{\renewcommand\@email{#1}}
\newcommand\@abstracttext{}\renewenvironment{abstract}{\global\setbox\@ewordabs\vbox\bgroup\hsize\textwidth
  \fontsize{9}{10.5}\selectfont\noindent\ignorespaces}{\egroup}
\newsavebox\@ewordabs
\newcommand\@keywords{}\newcommand\keywords[1]{\renewcommand\@keywords{#1}}
\renewcommand\maketitle{%
  {%
    \vspace*{0.1in}
    {\raggedright\fontsize{17}{20}\selectfont\bfseries\@title\par}
    \vspace{12pt}
    {\fontsize{13}{15}\selectfont\@author\textsuperscript{a,*}\par}
    \vspace{8pt}
    {\fontsize{8}{10}\selectfont\textsuperscript{a}\itshape\@affiliation\par}
    \vspace{6pt}\noindent\rule{\textwidth}{0.4pt}\par\vspace{6pt}
    {\fontsize{9}{10.5}\selectfont\bfseries Abstract\par}\vspace{4pt}
    \noindent\usebox\@ewordabs\par
    \vspace{8pt}
    {\fontsize{8}{10}\selectfont\noindent\textit{Keywords:} \@keywords\par}
    \vspace{6pt}\noindent\rule{\textwidth}{0.4pt}\par
    \vspace{10pt}}%
  \begingroup\renewcommand\thefootnote{*}%
  \footnotetext{\raggedright Corresponding author. \textit{E-mail address:} \@email}\endgroup
  \setcounter{footnote}{0}}
\renewcommand\footnoterule{\kern-3pt\hrule width 0.4\columnwidth\kern 2.6pt}
\pagestyle{plain}

\endinput
