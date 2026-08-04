# EEGEyeNet release metadata

Documentation-only audit; no candidate test signal or outcome was opened.

## Data  Structure Description.pdf

- Data Structure DescriptionsEEG(synchronised EEG) is a MATLAB structure thatcontains all the informationabout the current dataset (synchronized EEG and EyeTracking)
- Below the explanation of the sEEG substructures:
- sEEG.datacontinuous data matrix :
- sEEG.chanlocsstores information about the EEG channel locationsand channel names.
- Here,sEEG.chanlocsisa structurearrayoflength133(onerecordforeachofthe128channelsinthisdataset+1“Cz”centralreferenceelectrode+4eyechannels,as explained below).
- TheEYE-EEGtoolboxparses,imports,andsynchronizessimultaneouslyrecordedeye-trackingdata and adds it as extra channels to the EEG.
- sEEG.event
- TheEEGstructurefieldcontainsrecordsoftheexperimentaleventsthatoccurredwhile the data was being recorded, plus the user-definedevents.
- ● Type - contains the event type (e.g. L_saccade, L_ﬁxation,where Lcorresponds to the left eye)● latency - contains the event latency in data sampleunit● urevent - contains the index of the event in the originaltable).● Duration - the duration of the event● Endtime - endtime of the event● Sac_amplitude-forevents“L_saccade’’ and“R_saccade” only. Amplitudeofthe event (in degrees)● Sac_endpos_x- for events “L_saccade’’ and “R_saccade”only.● Sac_endpos_y - for events “L_saccade’’ and “R_saccade”only.● Sac_startpos_x- for events “L_saccade’’ and “R_saccade”only.● Sac_startpos_y- for events “L_saccade’’ and “R_saccade”only.● Sac_vmax-forevents“L_saccade’’ and“R_saccade”only.Maxvelocityduringexecution of the saccade.● Fix_avgpos_x-averagepositionduringthe“L_ﬁxation” or“R_ﬁxation” event,coordinate x.● Fix_avgpos_y-averagepositionduringthe“L_ﬁxation” or “R_ﬁxation” event,coordinate y.● Fix_avgpupilsize- averagepupilsize duringthe“L_ﬁxation” or “R_ﬁxation”event, coordinate x.
- Figure1. Part of the structure sEEg.event
- UR-EVENTS
- A separateeventstructure, sEEG.urevent, holdsalltheeventinformationoriginallyloaded into the dataset.
- EYE CHANNELS:
- Channel 130 - time
- Channel 131 - x coordinate of the eye
- Channel 132- y coordinate of the eye

## Experimental Paradigms.pdf

- TRIGGERS (in the EEG.event structure)
- The pro- and antisaccadeparadigmwas designedaccordingto the internationallystandardizedprotocolforantisaccadetesting.Eachtrialstartswithacentralfixationsquare.Theparticipantsareaskedtofocusonthecenterofthescreenforarandomizedtime-periodbetween1and3.5seconds.Subsequently, thecue(i.e.adot)appearshorizontallyontheleftortherighthand-sideofthecentralfixationsquare.Intheprosaccadetrials,theparticipantsareaskedtofocustheirgazeonthecueasfastaspossible,whileintheantisaccadetrialstheparticipantsareinstructedtoperformasaccadetowardstheoppositesideofthecue.Inbothcases,thecueisshownforonesecond.Assoonasthecuedisappears,theparticipantsshifttheirfocusbacktothecenterofthescreen.Datarecordedfollowingthisparadigmmaybeusedfor differentresearchpurposes,suchas estimatinggazedirectionor examiningresponses to inhibition.
- TRIGGERS (EEG.event structure)
- Participantsareaskedtofixateonaseriesofdotsthataresequentiallypresented,eachatoneof25differentscreenpositions.Unliketheothers,thedotatthecenterofthescreenappearsthreetimes,resultingin27trials(displayeddots)perblock,eachdotisdisplayedfor1.5to1.8seconds.Thepositionsofthedotswereselectedtoensurecoverageofallcornersofthescreenaswellasthecenter. Torecordalargernumberoftrialsandreducethepredictabilityof thesubsequentpositionsin theprimarysequenceof thestimulus,weusedifferentpseudo-randomizedorderingsof thedotspresentation,distributedin fiveexperimentalblocks,asshownin thefigurebelow. Theentireprocedureisrepeated6 timesduringthemeasurement, resulting in 810 stimuli for each participant.
- TRIGGERS (in the EEG.event structure)
- VisualSymbolSearch(VSS)isacomputerizedversionofaclinicalpediatricassessmenttomeasureprocessingspeed(SymbolSearchSubtestoftheWechslerIntelligenceScaleforChildrenIV. Participantsareshown15rowsat a time,whereeachrowconsistsoftwo\emph{target}symbols,fivesearchsymbolsandtwoadditionalsymbolsthatcontainrespectivelythewords``YES''and``NO''.Foreachrow, participantsneedtoindicatebyclickingwiththemousebuttononthe``YES''or``NO''symbol,whetherornotoneofthetwotargetsymbolsappearsamongthefivesearchsymbols.EachrecordingoftheVSSparadigm
- takes120secondswitha maximumof60trials,whereonetrialcorrespondstoonerow;in50%ofthetrialsoneofthetargetsymbolsdoesappearinthesearchsymbolsandintheremaining $50\%$ none does.Onceparticipantsfinisha setof15rows,theypressa``nextpage''buttonwhichdisplaysanewsetof15rows.Participantsareinstructedtosolveasmanyrows,ortrials,aspossiblewithin the given 120 seconds.Beforebeginningtheactualrecording,participantsperformatrainingoffourtrials,forwhichtheyreceivefeedbacktoensuretheyunderstandthetask.Nofeedbackisprovidedthroughouttheactualrecording.Datacollectedaccordingtothisparadigmmaybeusedforinvestigatingbehavioral and neurophysiological correlates of processingspeed

