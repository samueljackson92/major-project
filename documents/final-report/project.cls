% article.sty 23 Sep 85
% modified by F.W. Long to provide headers and footers 17 March 1989
% modified by F.W. Long for West Wales Software, 29 May 1989
% modified by J.J. Rowland for InFACT documents 7 July 1989
% modified by H.R.Nicholls for student project documents 14 November 1991
% modified by H.R.Nicholls to indicate UW Aberystwyth, and change postcode, 14.10.92

\typeout{Group project doc style modified by N.W. Hardy 2010-10-12}

% 11pt style - no option
% a4 style for the height, adjusted later for width
% converted to a class file  from projectstyle.sty by N.W.Hardy 2010-10-12
%   Added necessary header material
%   Removed unnecessary counters
%   Page and font size now from options on the the loaded style
%   Added a subtitle
%   Removed cic and coconf
%   Moved the front page address block to new position
%   Made the header just a line not a table (cic and coconf)
%   Changed all to Aberystywth University

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{project}[2010/10/12 Group project Class]

\LoadClass[11pt,twoside,a4paper]{article}


\def\labelenumi{\arabic{enumi}.} 
\def\theenumi{\arabic{enumi}} 
\def\labelenumii{(\alph{enumii})}
\def\theenumii{\alph{enumii}}
\def\p@enumii{\theenumi}
\def\labelenumiii{\roman{enumiii}.}
\def\theenumiii{\roman{enumiii}}
\def\p@enumiii{\theenumi(\theenumii)}
\def\labelenumiv{\Alph{enumiv}.}
\def\theenumiv{\Alph{enumiv}} 
\def\p@enumiv{\p@enumiii\theenumiii}

\def\labelitemi{$\bullet$}
\def\labelitemii{\bf --}
\def\labelitemiii{$\ast$}
\def\labelitemiv{$\cdot$}

\def\verse{\let\\=\@centercr 
 \list{}{\itemsep\z@ \itemindent -1.5em\listparindent \itemindent 
 \rightmargin\leftmargin\advance\leftmargin 1.5em}\item[]}
\let\endverse\endlist
\def\quotation{\list{}{\listparindent 1.5em
 \itemindent\listparindent
 \rightmargin\leftmargin \parsep 0pt plus 1pt}\item[]}
\let\endquotation=\endlist
\def\quote{\list{}{\rightmargin\leftmargin}\item[]}
\let\endquote=\endlist

\def\descriptionlabel#1{\hspace\labelsep \bf #1}
\def\description{\list{}{\labelwidth\z@ \itemindent-\leftmargin
 \let\makelabel\descriptionlabel}}
\let\enddescription\endlist


\def\theequation{\arabic{equation}}


\def\titlepage{\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
 \else \newpage \fi \thispagestyle{empty}\c@page\z@}
\def\endtitlepage{\if@restonecol\twocolumn \else \newpage \fi}

\arraycolsep 5pt \tabcolsep 6pt \arrayrulewidth .4pt \doublerulesep 2pt 
\tabbingsep \labelsep 

\skip\@mpfootins = \skip\footins
\fboxsep = 3pt \fboxrule = .4pt 


\def\thepart{\Roman{part}} \def\thesection {\arabic{section}}
\def\thesubsection {\thesection.\arabic{subsection}}
\def\thesubsubsection {\thesubsection .\arabic{subsubsection}}
\def\theparagraph {\thesubsubsection.\arabic{paragraph}}
\def\thesubparagraph {\theparagraph.\arabic{subparagraph}}

\def\@pnumwidth{1.55em}
\def\@tocrmarg {2.55em}
\def\@dotsep{4.5}
\setcounter{tocdepth}{3}

% set parskip to zero locally within the lists

\def\tableofcontents{\section*{CONTENTS}
 {\parskip=0pt \@starttoc{toc}}}
\def\l@part#1#2{\addpenalty{\@secpenalty}
 \addvspace{2.25em plus 1pt} \begingroup
 \@tempdima 3em \parindent \z@ \rightskip \@pnumwidth \parfillskip
-\@pnumwidth 
 {\large \bf \leavevmode #1\hfil \hbox to\@pnumwidth{\hss #2}}\par
 \nobreak \endgroup}
\def\l@section#1#2{\addpenalty{\@secpenalty} \addvspace{1.0em plus 1pt}
\@tempdima 1.5em \begingroup
 \parindent \z@ \rightskip \@pnumwidth 
 \parfillskip -\@pnumwidth 
 \bf \leavevmode #1\hfil \hbox to\@pnumwidth{\hss #2}\par
 \endgroup}
\def\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}
\def\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}
\def\l@paragraph{\@dottedtocline{4}{7.0em}{4.1em}}
\def\l@subparagraph{\@dottedtocline{5}{10em}{5em}}
\def\listoffigures{\section*{LIST OF FIGURES}
 {\parskip=0pt \@starttoc{lof}}}
\def\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\def\listoftables{\section*{LIST OF TABLES}
 {\parskip=0pt \@starttoc{lot}}}
\let\l@table\l@figure


\def\thebibliography#1{\section*{REFERENCES}\list
 {[\arabic{enumi}]}{\settowidth\labelwidth{[#1]}\leftmargin\labelwidth
 \advance\leftmargin\labelsep
 \usecounter{enumi}}
 \def\newblock{\hskip .11em plus .33em minus -.07em}
 \sloppy
 \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist
 

\newif\if@restonecol
\def\theindex{\@restonecoltrue\if@twocolumn\@restonecolfalse\fi
\columnseprule \z@
\columnsep 35pt\twocolumn[\section*{Index}]
 \markboth{INDEX}{INDEX}\thispagestyle{plain}\parindent\z@
 \parskip\z@ plus .3pt\relax\let\item\@idxitem}
\def\@idxitem{\par\hangindent 40pt}
\def\subitem{\par\hangindent 40pt \hspace*{20pt}}
\def\subsubitem{\par\hangindent 40pt \hspace*{30pt}}
\def\endtheindex{\if@restonecol\onecolumn\else\clearpage\fi}
\def\indexspace{\par \vskip 10pt plus 5pt minus 3pt\relax}

\def\footnoterule{\kern-3\p@ 
 \hrule width .4\columnwidth 
 \kern 2.6\p@} 

\long\def\@makefntext#1{\parindent 1em\noindent 
 \hbox to 1.8em{\hss$^{\@thefnmark}$}#1}


\setcounter{topnumber}{2}
\def\topfraction{.7}
\setcounter{bottomnumber}{1}
\def\bottomfraction{.3}
\setcounter{totalnumber}{3}
\def\textfraction{.2}
\def\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\def\dbltopfraction{.7}
\def\dblfloatpagefraction{.5}

\long\def\@makecaption#1#2{
 \vskip 10pt 
 \setbox\@tempboxa\hbox{#1: #2}
 \ifdim \wd\@tempboxa >\hsize \unhbox\@tempboxa\par \else \hbox
to\hsize{\hfil\box\@tempboxa\hfil} 
 \fi}


\def\thefigure{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{Figure \thefigure}
\def\figure{\@float{figure}}
\let\endfigure\end@float
\@namedef{figure*}{\@dblfloat{figure}}
\@namedef{endfigure*}{\end@dblfloat}

\def\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{Table \thetable}
\def\table{\@float{table}}
\let\endtable\end@float
\@namedef{table*}{\@dblfloat{table}}
\@namedef{endtable*}{\end@dblfloat}


% set up the information for the title page
% title redefined to also set shorttitle

\def\title#1{\gdef\@title{#1} \gdef\@shorttitle{#1}}
\def\subtitle#1{\gdef\@subtitle{#1}}
\def\shorttitle#1{\gdef\@shorttitle{#1}}
\def\client#1{\gdef\@client{{\it Client:} & #1 \\}} \gdef\@client{}
\def\configref#1{\gdef\@configref{#1}}
\def\projref#1{\gdef\@projref{{\it Proj. Ref.:} & #1 \\}} \gdef\@projref{}
\def\version#1{\gdef\@version{#1}}
\def\status#1{\gdef\@status{#1}}


% note pagestyle set to myheadings after the title page

\def\maketitle{\par
 \begingroup
 \def\thefootnote{\fnsymbol{footnote}}
 \def\@makefnmark{\hbox 
 to 0pt{$^{\@thefnmark}$\hss}} 
 \if@twocolumn 
 \twocolumn[\@maketitle] 
 \else \newpage
 \global\@topnum\z@ \@maketitle \fi\thispagestyle{myheadings}\@thanks
 \endgroup
 \setcounter{footnote}{0}
 \let\maketitle\relax
 \let\@maketitle\relax
 \gdef\@thanks{}\gdef\@author{}\gdef\@title{}\let\thanks\relax}

% generate the title page
% note the extra {...} to restrict the scope of headheight

\def\@maketitle{{\headheight 12pt\newpage
\thispagestyle{empty}\c@page\z@
 \null
 \vskip 0.4in
 \hskip 0.3in
\begin{minipage}{5.2in}
 \centering\Large\bf\@title\par
 \centering\large\bf\@subtitle
\end{minipage}

 \hskip 0.3in
\begin{tabular}[t]{lp{3.3in}}
{\it Author:}        & \@author  \\
{\it Email:}        & wde7@aber.ac.uk \& csf7@aber.ac.uk  \\
{\it Degree Scheme:}        & G601 Software Engineering MEng  \\
\@projref
\@client
{\it Config.\ Ref.:} & \@configref  \\
{\it Date:}          & \@date    \\
{\it Version:}       & \@version \\
{\it Status:}        & \@status  \\
\end{tabular}\par
\vfill
\hskip 0.3in
\begin{minipage}{5.2in}
\begin{tabular}{l}
Department of Computer Science, \\
Aberystwyth University,\\
Aberystwyth, \\
Ceredigion, SY23 3DB, \\
U.K. \\
 \\
\copyright Aberystwyth University \number\year \\
\end{tabular}
\end{minipage}
\newpage}}

\def\abstract{\if@twocolumn
\section*{Abstract}
\else \small 
\begin{center}
{\bf Abstract\vspace{-.5em}\vspace{0pt}} 
\end{center}
\quotation 
\fi}
\def\endabstract{\if@twocolumn\else\endquotation\fi}

\mark{{}{}} 

\if@twoside \def\ps@headings{\def\@oddfoot{}\def\@evenfoot{}\def\@evenhead{\rm
\thepage\hfil \sl \leftmark}\def\@oddhead{\hbox{}\sl \rightmark \hfil
\rm\thepage}\def\sectionmark##1{\markboth {\uppercase{\ifnum \c@secnumdepth
>\z@
 \thesection\hskip 1em\relax \fi ##1}}{}}\def\subsectionmark##1{\markright
{\ifnum \c@secnumdepth >\@ne
 \thesubsection\hskip 1em\relax \fi ##1}}}
\else \def\ps@headings{\def\@oddfoot{}\def\@evenfoot{}\def\@oddhead{\hbox
{}\sl \rightmark \hfil \rm\thepage}\def\sectionmark##1{\markright
{\uppercase{\ifnum \c@secnumdepth >\z@
 \thesection\hskip 1em\relax \fi ##1}}}}
\fi

% set up my own headers and footers

\def\ps@myheadings{\def\@oddhead{\headline}\def\@oddfoot{\footline}
\def\@evenhead{\headline}\def\@evenfoot{\footline}
\def\sectionmark##1{}\def\subsectionmark##1{}}

\newcommand{\headline}%
{\it \@shorttitle/\@version(\@status)\hfill ~}
\newcommand{\footline}%
{{\it Aberystwyth University/Computer Science} \hfill Page \thepage\ of \pageref{thelastpage}}

% set up our style parameters

\hyphenpenalty=5000
\doublehyphendemerits=10000
\finalhyphendemerits=10000

\topsep 0pt
\parskip 7pt plus 2pt minus 1pt
\parindent 0pt

\setlength{\textwidth}{5.5in}
\setlength{\evensidemargin}{\oddsidemargin}

% override the style for section and subsubsubsection headings

\def\section{\@startsection {section}{1}{\z@}{-3.5ex plus -1ex minus
 -.2ex}{2.3ex plus .2ex}{\large\bf}}
\def\paragraph{\@startsection
 {paragraph}{4}{\z@}{-3ex plus -1ex minus -.2ex}{1ex plus .2ex}{\normalsize\bf}}

% subsubsubsection headings to be numbered, set secnumdepth accordingly

\setcounter{secnumdepth}{4}

% redefine today to produce English format dates

\def\today{\number\day\space\ifcase\month\or
 January\or February\or March\or April\or May\or June\or
 July\or August\or September\or October\or November\or December\fi
 \space\number\year}

% more style definitions

\ps@myheadings \pagenumbering{arabic}
\onecolumn \if@twoside\else\raggedbottom\fi 
