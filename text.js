class TextSource {

    constructor(identifier){
        if (!identifier) {
            throw new Error('Text identifier is not provided.');
            }

        this.identifier_ = identifier;
        this.getText();
    }

    /**
    * Get data identifier.
    * @returns {string} The data identifier.
    */
    dataIdentifier() {
    return this.dataIdentifier_;
    }

    /**
    * Get source text.
    * @return {string} The whole source text corpus.
    */
    getText() {
    const text = `          Oliver Twist

                                  von

                            Charles Dickens

                  mit einer biographischen Einleitung
                          von Johannes Gaulke

                     Globus Verlag GmbH Berlin

Unter den großen Humoristen des vorigen Jahrhunderts, die zugleich
Tendenzschriftsteller im besten Sinne waren, nimmt Charles Dickens
einen hervorragenden Platz ein, den er trotz des schnellen Wandels des
literarischen Geschmacks und der Kunstanschauung in der Weltliteratur
behaupten wird. Dickens ist nicht nur der Lieblingsdichter seines
Volkes, sondern er ist schon zu Lebzeiten in allen Ländern des
Erdenrunds heimisch geworden. In Hütte und Palast sind seine Werke
gedrungen und haben überall starke und nachhaltige Wirkungen ausgeübt.
Begabt mit dem köstlichen Humor, der mit dem einen Auge weint und dem
anderen lacht, ist Dickens allen denen, die auf der Höhe des Lebens
wandeln, ein treuer Mentor geworden, den Elenden und Enterbten des
Lebensglücks aber ein aufrichtiger Freund und Tröster.

Charles Dickens konnte zum ganzen Volke von allen den Dingen, die
unsre Welt ausmachen, sprechen, weil er das Leben gründlich kannte,
weil er selbst alle Wechselfälle des Lebens an sich selbst erfahren
hatte. Als Kind wenig bemittelter Eltern am 7. Februar 1812 in Landport
bei Portsmouth geboren, mußte er schon im Alter von zehn Jahren,
als sein Vater in London ins Schuldgefängnis gewandert war, für den
eigenen Lebensunterhalt sorgen. Während er als Laufbursche gegen einen
kärglichen Wochenlohn tätig war, vernachlässigte er naturgemäß seine
Schulbildung gänzlich, und er genoß erst, nachdem der Vater eine
bescheidene Stellung in London erlangt hatte, als zwölfjähriger Knabe
einen besseren Unterricht. Den Mangel eines systematischen Unterrichts
hat er durch Selbstunterricht, der sich auf alle Gebiete des
Wissens erstreckte, namentlich aber durch das Studium der englischen
Schriftsteller ausgeglichen. Im Jahre 1833 veröffentlichte er, nachdem
er sich schon als Journalist an führenden Blättern unter dem Pseudonym
Boz mit großem Erfolge betätigt hatte, sein erstes Buch, eine Reihe von
Skizzen aus dem Londoner Volksleben in zwei Bänden. Einige Jahre später
folgten die «_Pickwick papers_», die ihn mit einem Schlage zu einem
gelesenen und in allen Schichten gleich geschätzten Autor machten.
Das Buch, das in einer Reihe von lose aneinandergefügten Skizzen die
Abenteuer einiger Mitglieder des Pickwickklubs auf ihrer Reise durch
England schildert, enthält in gewissem Sinne das Programm des späteren
Dickens, der das Leben schildert, wie es sich ihm darbietet, immer
von dem Gedanken getragen, moralische Wirkungen zu erzielen und den
Menschen mit seiner Umwelt zu versöhnen. Um dieses Ziel zu erreichen,
schrickt er nicht vor Übertreibungen eines Zustandes oder einer
Handlung zurück und macht selbst, um möglichst eindringlich zu wirken,
seine Figuren, die meistens sehr lebensvoll einsetzen, zu menschlichen
Karikaturen.

In rascher Folge erscheinen in den dreißiger und vierziger Jahren
des vorigen Jahrhunderts die Hauptwerke Dickens. Die Reihe eröffnet
«Oliver Twist» (1838), das als das erste realistische, aus dem
Volkstum geschöpfte Buch mit außerordentlichem Enthusiasmus in England
aufgenommen wurde und bald seinen Weg über den Erdball machte.
Es folgten: «Nicholas Nickleby» (1839) und «_Master Humphrey's
clock_» (1840), ein Werk, das sich ähnlich wie die «Pickwickier»
aus Einzelerzählungen zusammensetzt, sich aber vor einem ernsteren
Hintergrund abspielt und tiefergreifende Menschenschicksale darstellt.

In den vierziger Jahren unternahm Dickens, der inzwischen zu einem
gewissen Wohlstand gelangt war, große Auslandsreisen. Die Hauptfrucht
seiner ersten Amerikareise (1842) ist der Roman «Martin Chuzzlewit»,
in dem er die Heuchelei der Amerikaner mit scharfen Hieben geißelt.
Auch in seinen «_American notes_» läßt er es an harten Bemerkungen
über die Amerikaner und amerikanischen Einrichtungen nicht fehlen. Die
Amerikaner haben ihm die geringe Meinung über sie und ihr Land, der
er zu wiederholten Malen Ausdruck gegeben hat, nicht nachgetragen,
sondern ihm in Neuyork, Chicago und anderen Städten prächtige Denkmäler
errichtet.

In Italien schrieb Dickens den Roman «Chimes» (1844), am Genfer
See «_Battle of Life_» (1846). Fast gleichzeitig entstand «_Dombey
and son_», ein Lebensbild aus dem Bürgertum, in dem Episoden von
ergreifender Tragik und grotesker Komik einander folgen. Auf der Höhe
des Schaffens stehend, schrieb Dickens Ende der vierziger Jahre den
autobiographischen Roman «David Copperfield», der nach Plan und Anlage
als ein wahrhaft geniales Werk genannt zu werden verdient. In der
Charakterisierung der Person hat Dickens hier die höchste Meisterschaft
erreicht, auch ist die Handlung einheitlicher und geschlossener als in
den Werken seiner ersten Periode.

David Copperfield ist wie die meisten Romane ein sozialer Tendenzroman.
Für Dickens, der aus dem Volke hervorgegangen war, der auch als
Dichter ein Selfmademan war, war die Kunst immer nur ein Mittel zum
Zweck, nicht Selbstzweck, wie es eine spätere französische Richtung
durch den Grundsatz «_l'art pour l'art_» ausdrückt. Dickens ist
daher keiner begrenzten Gruppe oder Kunstrichtung einzureihen; er
ist weder Realist noch Idealist im herkömmlichen Sinne, sondern auch
als Künstler immer nur Moralist. Zwar sind die Zustände stets mit
den Augen des Realisten gesehen, er ist sogar ein Kleinmaler von
einer Prägnanz des Ausdrucks wie wenige, aber darüber hinaus reicht
sein Wirklichkeitssinn nicht. Sobald er an den Menschen herantritt,
versagt sein Charakterisierungsvermögen, er schildert die Menschen
nicht wie sie sind, aus dem Milieu heraus, sondern wie er wünscht, daß
sie sein möchten. Nur selten gelingt es ihm, einen der Wirklichkeit
entsprechenden Menschen zu zeichnen; seine Romanfiguren sind entweder
idealisiert oder karikiert -- im besten Falle Typen, keine Individuen.
Entweder sind sie Erzbösewichter oder herzensgute Engel. Und zum Schluß
erhalten sie alle, ganz im Einklang mit dem höchsten moralischen
Grundgesetz, ihre Strafe oder ihre Belohnung für das, was sie getan
oder unterlassen haben.

Am besten gelingen Dickens die Gestalten aus dem Volk, mit ihnen ist
der Dichter aufgewachsen, mit ihnen hat er gelitten, mit ihnen kann er
daher auch empfinden. Auch in die Seele des Kindes vermag sich Dickens
zu versetzen; hier wirkt sein Pathos immer echt, ob er das Elend des
ausgesetzten Kindes schildert, die Qualen und Entbehrungen eines
kleinen Bettlers oder gar den Tod eines unglücklichen kleinen Wesens.

Je weiter sich Dickens vom Volkstum entfernt, umso unklarer und
verschwommener werden seine Gestalten, doch weiß er auch hier
wiederum mit glücklichem Griff das Milieu, in dem eine Lordschaft
oder gar ein englischer Herzog sich bewegt, festzuhalten. Man
sieht gern über die angedeuteten Schwächen hinweg, da der Dichter
unerschöpflich in der Erfindung komischer und grotesker Situationen
ist und mit einem von Herzen kommenden und zu Herzen gehenden Humor
alle menschlichen Schwächen und Verirrungen zu entschuldigen weiß.
Selbst dem tiefgesunkenen Verbrecher haftet immer noch ein menschlich
liebenswürdiger Zug an. Ohne gerade Kriminalpsychologe zu sein,
schildert Dickens seine Gestalten fast durchgängig als Produkte ihrer
Umgebung und behandelt auch den schändlichsten Missetäter mit Nachsicht
und Milde. So nur konnte er zu einem Anwalt der Unglücklichen und
Enterbten werden.

In der zweiten Periode seines dichterischen Schaffens, die die beiden
letzten Jahrzehnte seines Lebens umfaßt, treten die Eigenarten und
Schwächen des Dichters immer schärfer hervor. Rastlos tätig, lockert
sich in seinen Romanen immer mehr die Komposition, auf langatmige
Schilderungen folgen knappe dramatische Evolutionen und spannende
Konflikte, die zu einem plötzlichen Abschluß drängen. Besonders
charakteristisch ist in dieser Beziehung der vierbändige Roman «_Our
mutual friend_», aber auch «_Bleakhouse_» und «_Tale of two cities_»,
wo die französische Revolution den Hintergrund bildet, lassen die
Einheitlichkeit des Plans vermissen.

Charles Dickens war während seines ganzen Lebens von einem
Arbeitseifer, der weder Rast noch Ruh kennt, beseelt. Während er seine
großen Romane schrieb, war er im Nebenfach als Journalist und Redakteur
tätig. Im Jahre 1845 trat er in die Redaktion der neubegründeten
Zeitung «_Daily News_», die auch seine italienischen Reisebilder zuerst
veröffentlichte, ein. 1849 gab er eine Wochenschrift «_Household
Words_», die der Unterhaltung und Belehrung diente, heraus. Daneben
fand er Zeit zu Vortragsreisen in England, Irland und Amerika, die ihm
Reichtümer und hohe Ehrungen einbrachten, aber auch die mittelbare
Ursache zu seinem plötzlichen Tode wurden. Er starb, vom Schlage
getroffen, nach kurzem Krankenlager auf seinem Landsitz Gadshill,
am 9. Juli 1870 im Alter von 58 Jahren. Seine Gebeine wurden in der
Westminsterabtei, dem Pantheon Englands, beigesetzt.

Wenden wir uns nunmehr der in diesem Bande veröffentlichten Erzählung
«Oliver Twist» zu, so werden wir die Vorzüge und Schwächen Dickensscher
Erzählungskunst gerade an diesem Werke höchst eindringlich wahrnehmen
können. Oliver Twist ist Dickens hervorragendstes Jugendwerk und
behandelt die Geschichte einer Jugend. Zweifellos haben eigene
Jugendeindrücke dem Dichter die Direktive zu dieser Arbeit gegeben.
Wie der kleine Oliver, so hat auch Dickens, zwar unter anderen
Verhältnissen, aber ebenso mühselig, sich emporringen müssen. Das
Leben hatte den Dichter schon in zarter Jugend hart angepackt, aber
wie das Gold sich im Feuer läutert, so läutert sich die Seele im
Lebenskampf, der Schmutz haftet nur dem Schmutzigen an, wer gesund und
rein empfindet, muß schließlich alle Widerwärtigkeiten des Lebens
überwinden. Das ist der Leitgedanke in Oliver Twist. Höchst drastische
Bilder läßt der Dichter vor unserem geistigen Auge entstehen, scharf
zugespitzte Situationen schildert er mit einer Anschaulichkeit, die uns
um das Schicksal des jugendlichen Helden mit banger Sorge erfüllt. Wir
empfinden und fühlen mit Oliver Twist, wir fürchten gar um sein Leben
und zittern um sein Seelenheil. Oftmals hat es den Anschein, als müsse
die Katastrophe jäh über ihn hereinbrechen, aber immer wieder entwirren
sich die verworrenen Schicksalsfäden, bis ihm endlich die Erlösung aus
unwürdigen Zuständen, in die er ohne seine Schuld geraten ist, wird.

Wenn man den moralischen Maßstab an eine dichterische Arbeit anlegen
will, so vollzieht sich in «Oliver Twist» alles ganz folgerichtig:
die Tugend muß schließlich siegen, denn so will es die moralische
Weltordnung. Vom literarischen Gesichtspunkt betrachtet, ließe sich
allerdings mancherlei gegen den Optimismus Dickens einwenden; man
merkt gar zu schnell die moralisierende Absicht und wird verstimmt.
Dagegen kann Dickens als Zustandsschilderer auch hier vor jeder
literarischen Kritik bestehen. Wie anschaulich sind allein die
Verbrecherschlupfwinkel geschildert! Wie überzeugend die Örtlichkeiten
des dunkelsten Londons! Man gewinnt hier überall den Eindruck des
Selbstgeschauten. Dickens bedient sich zur Erreichung seines Zwecks
oft ungewöhnlicher Mittel und verblüffender Wendungen. Er konstruiert
die unwahrscheinlichsten Situationen und nimmt es auch mit den
Tatsachen nicht so genau, um eine Kontrastwirkung zu erzielen. Einzelne
Begebenheiten streifen fast das Niveau des Kolportageromans, während
andere den Eindruck höchster Künstlerschaft auf den Leser machen.

«Oliver Twist» ist eine Arbeit, die nicht mit dem Kopf, sondern mit
dem Herzen geschrieben ist. Es ist der Roman des Kindes, vielleicht
der erste dieser Art in der neueren Literatur. Der maßlose Jammer der
ausgesetzten und verlassenen Kinder, von denen es im heutigen London
noch hunderte und aberhunderte gibt, hat den Dichter angeregt zu einer
Arbeit, die ein Appell an die Welt zur Abhilfe der verrotteten Zustände
sein soll. Wir leben im «Jahrhundert des Kindes»! Männer und Frauen
aller Kreise haben zusammengewirkt, um eine Hebung des sittlichen
Niveaus, aber auch der materiellen Lage der Kinder der Ärmsten zu
erzielen. Der Dichter des «Oliver Twist» verdient als Vorläufer dieser
Bewegung bezeichnet zu werden. Menschengüte und Kinderliebe sprechen
aus jeder Zeile des Buches; ohne diese Qualitäten hätte es schwerlich
seinen Platz in der Weltliteratur behauptet.

Von dem feinen Verständnis der Kinderseele und den Bedürfnissen des
Kindes legt neben «Oliver Twist» auch Dickens «_A child's history of
England_» ein glänzendes Zeugnis ab. In einer, den Anschauungskreis des
Kindes angemessenen Form schildert Dickens hier die Hauptereignisse der
englischen Geschichte mit höchster Eindringlichkeit und Wahrhaftigkeit,
aber auch zugleich frei von jeder aufdringlichen chauvinistischen
Tendenz. Das Buch, das in England und Amerika zu den meistgelesenen
Büchern zählt, hat bei uns bei weitem nicht die ihm gebührende
Beachtung gefunden, obgleich wir in der geschichtlichen Jugendliteratur
ihm nichts Ebenbürtiges zur Seite stellen können.

Charles Dickens hat in England viele Nachahmer gefunden, aber niemand
hat ihn, dem das Schreiben eine sittliche Aufgabe war, auch nur im
entferntesten erreicht. Seine Nachahmer haben ihn eigentlich nur
in der Breite der Anlage und der Umständlichkeit der Schilderung
getroffen, nicht aber in der Eindringlichkeit seines Vortrags und
in der Überzeugungskraft, mit der er seine Tendenz verficht. Bald
wird ein halbes Jahrhundert seit seinem Tode verflossen sein, die
englische Familienblattliteratur hat inzwischen hunderttausende von
Neuerscheinungen auf den Markt geworfen, aber doch ist Dickens der
Dichter des Volkes geblieben. Von den großen Humoristen des vorigen
Jahrhunderts kann ihm nur einer als gleichwertig zur Seite gestellt
werden: unser Fritz Reuter, der ja auch wie Dickens in einer harten
Schule der Entbehrungen zum Dichter und Menschenfreund herangewachsen
ist. Sie liebten beide das Volk, weil sie es als echte Söhne des Volkes
genau kannten, und sie haben beide dadurch, daß sie die erbärmliche
Alltäglichkeit mit echter Poesie und echtem Humor durchtränkt haben,
Millionen entzückt und mit dem Leben versöhnt. Wer das zuwege bringt,
ist ein Wohltäter der Menschheit. In diesem Sinne hat sich einer der
größten Staatsmänner Englands, Gladstone, über Dickens geäußert; auch
er feierte ihn als einen Erzieher und Wohltäter seines Volkes. Charles
Dickens wird daher, mag er einer strengen literarischen Kritik auch
nicht immer standhalten, mag er selbst als Menschenschilderer überholt
worden sein, dennoch weiterleben und noch viele Generationen durch
seinen köstlichen Humor entzücken und begeistern.

                                                       Johannes Gaulke.




1. Kapitel.

    Handelt von dem Orte, an dem Oliver Twist geboren wurde, und von
    den seine Geburt begleitenden Umständen.


Eine Stadt, die ich aus gewissen Gründen nicht näher bezeichnen will,
der ich aber auch keinen erdichteten Namen beilegen möchte, besitzt
unter anderen öffentlichen Gebäuden gleich den meisten anderen Städten,
sie mögen groß oder klein sein, von alters her ein Armenhaus, und in
diesem wurde an einem Tage, dessen genaues Datum für den Leser kein
besonderes Interesse hat, das Mitglied der sterblichen Menschheit
geboren, dessen Name in der Überschrift dieses Kapitels angegeben ist.

Lange Zeit, nachdem der Wundarzt des Kirchspiels ihn in diese Welt
der Mühen und Sorgen befördert hatte, blieb es äußerst zweifelhaft,
ob er lange genug leben würde, um überhaupt eines Namens zu bedürfen.
Es war nämlich tatsächlich mit erheblichen Schwierigkeiten verbunden,
Oliver dahin zu bringen, daß er sich der Aufgabe, Atem zu holen,
selbst unterzog -- einem mühsamen Geschäfte, das die Gewohnheit uns
aber freilich zu einer notwendigen Lebensbedingung gemacht hat; eine
Zeitlang lag er nach Luft schnappend auf einer kleinen Matratze
aus Schafwolle und schien sich in der Schwebe zwischen dieser und
jener Welt zu befinden, wobei die Wage sich entschieden zugunsten
der letzteren neigte. Wenn Oliver während dieser kurzen Zeit von
sorglichen Großmüttern, geschäftigen Tanten, erfahrenen Wärterinnen
und hochgelahrten Doktoren umgeben gewesen wäre, so würde er natürlich
die Stunde nicht überlebt haben, allein es war niemand in seiner
Nähe, außer einer alten Frau, die sich infolge des ungewohnten
Genusses von Bier in einer etwas angeheiterten Stimmung befand, und
dem Kirchspielwundarzte, der die Geburtshilfe kontraktmäßig leistete.
Oliver und die Natur fochten also die Sache zwischen sich ganz
allein aus, und die Folge davon war, daß nach kurzem Kampfe Oliver
atmete, nieste und endlich den Insassen des Armenhauses die Tatsache
ankündigte, daß dem Kirchspiele eine neue Last aufgebürdet worden sei,
indem er ein so lautes Geschrei erhob, wie man es füglicherweise von
einem neugeborenen Knaben erwarten konnte.

Als Oliver diesen ersten Beweis von der freien und selbständigen
Tätigkeit seiner Lungen gab, bewegte sich die geflickte Decke, die
nachlässig über die eiserne Bettstelle gebreitet war; das bleiche
Antlitz einer jungen Frau erhob sich matt von dem harten Pfühle, und
eine schwache Stimme brachte mühsam die Worte hervor: «Lassen Sie mich
das Kind sehen, dann will ich gern sterben.»

Der Wundarzt, der vor dem Kamine saß und seine Hände abwechselnd an dem
Feuer wärmte und rieb, erhob sich bei den Worten der jungen Frau, trat
an das Kopfende des Bettes und sagte mit mehr Freundlichkeit im Tone,
als man ihm zugetraut haben würde: «Oh, Sie dürfen jetzt nicht vom
Sterben sprechen.»

«Der Herr segne ihr gutes Herzchen, nein!» unterbrach ihn die Wärterin,
indem sie eine grüne Glasflasche, von deren Inhalt sie in einer
verschwiegenen Ecke mit sichtlichem Behagen gekostet hatte, rasch in
die Tasche steckte. «Der Herr segne ihr gutes Herzchen; wenn sie erst
so alt geworden ist wie ich und dreizehn Kinder gehabt hat und alle
sind tot bis auf zwei, die zusammen mit mir im Armenhause sind, so wird
sie schon auf andere und vernünftigere Gedanken kommen; der Herr segne
ihr gutes Herzchen. Bedenken Sie nur, Frauchen, was es heißt, Mutter
eines so süßen, kleinen Lämmchens zu sein.»

Diese tröstlichen Worte schienen ihre Wirkung zu verfehlen. Die
Wöchnerin schüttelte den Kopf und streckte die Arme nach dem Kinde
aus. Der Wundarzt reichte es ihr, sie küßte es, heftig erregt, mit
den kalten, weißen Lippen auf die Stirn, fuhr mit den Händen über ihr
Gesicht, blickte wild umher, schauderte, sank zurück -- und starb.

«'s ist aus mit ihr», sagte der Wundarzt nach einigen Bemühungen, sie
wieder zum Leben zurückzubringen.

«Das arme Kind!» sagte die Wärterin, indem sie den Pfropfen der
grünen Flasche aufhob, der auf das Kissen gefallen war, als sie sich
niederbeugte, um das Kind aufzunehmen. «Armes Kind!»

«Sie brauchen nicht zu mir zu schicken, wenn das Kind schreit», fuhr
der Wundarzt fort, während er kaltblütig seine Handschuhe anzog. «Es
wird wahrscheinlich sehr unruhig sein; geben Sie ihm dann ein wenig
Hafergrütze.» Er setzte den Hut auf, trat noch einmal an das Bett und
sagte: «Die Mutter sah gut aus; woher kam sie?»

«Sie wurde gestern abend gebracht,» erwiderte die Wärterin, «auf Befehl
des Direktors. Man hatte sie auf der Straße liegend gefunden, und
sie muß ziemlich weit hergewandert sein, denn ihre Schuhe waren ganz
zerrissen; aber woher sie kam, oder wohin sie wollte, das weiß niemand.»

Der Wundarzt beugte sich über die Verblichene, hob die rechte Hand
derselben empor und bemerkte kopfschüttelnd: «Die alte Geschichte; kein
Trauring, wie ich sehe. Hm! gute Nacht!»

Er ging zu seinem Abendessen, und die Wärterin setzte sich, nachdem
sie sich noch einmal an der grünen Flasche erlabt hatte, auf einen
Stuhl in der Nähe des Feuers und begann das Kind anzukleiden.
Bis zu diesem Augenblick hätte man nicht sagen können, ob es das
Kind eines Edelmannes oder eines Bettlers sei; das dürftige,
verwaschene Kinderzeug des Armenhauses bezeichnete indes sogleich
seine gegenwärtige und zukünftige Stellung in der Welt, sein ganzes
Schicksal, als Kirchspielkind -- Waise des Armenhauses, halb verhungert
und unter Mühe und Plackerei, verachtet von allen, bemitleidet von
niemand, durch die Welt geknufft und gestoßen zu werden.

Oliver schrie mit kräftiger Stimme; hätte er wissen können, daß er eine
Waise war, überliefert der zärtlichen Fürsorge von Kirchenältesten und
Kirchenvorstehern, so würde er vielleicht noch lauter geschrien haben.




2. Kapitel.

    Handelt von Oliver Twists Heranwachsen und kümmerlicher Ernährung
    sowie von einer Sitzung des Armenkollegiums.


Während der nächsten acht bis zehn Monate war Oliver das Opfer einer
systematischen Gaunerei und Betrügerei. Er wurde aufgepäppelt. Die
elende und verlassene Lage der kleinen Waise wurde von der Behörde
des Armenhauses pflichtschuldigst der des Kirchspiels gemeldet. Die
letztere forderte von der ersteren würdevoll einen Bericht darüber ab,
ob sich nicht in «dem Hause» eine Frauensperson befände, die dem Kinde
seine natürliche Nahrung reichen könnte. Die Behörde des Armenhauses
beantwortete die Anfrage untertänigst mit nein, und daraufhin faßte
die Kirchspielbehörde den hochherzigen Entschluß, Oliver in ein etwa
drei Meilen entferntes Filialarmenhaus bringen zu lassen, wo zwanzig
bis dreißig andere kleine Übertreter der Armengesetze unter der
mütterlichen Aufsicht einer ältlichen Frau, welche für jeden derselben
wöchentlich sieben und einen halben Penny erhielt, aufwuchsen, ohne
zu gut genährt oder zu warm gekleidet und verzärtelt zu werden. Mit
sieben und einem halben Penny läßt sich nicht viel beschaffen, und die
Matrone war klug und erfahren. Sie wußte, wie leicht sich Kinder den
Magen überladen können und was ihnen dient, ebenso genau aber auch, was
ihr selbst gut war; sie verwendete daher einen beträchtlichen Teil des
für die Kinder Bestimmten in ihrem eigenen Nutzen, fand demnach in der
tiefsten noch eine tiefere Tiefe und bewies somit, daß sie es in der
Experimentalphilosophie wirklich weit gebracht hatte.

Jedermann kennt die Geschichte eines anderen Experimentalphilosophen,
nach dessen ruhmwürdiger Theorie ein Pferd imstande war, ohne Nahrung
zu leben, und der jene so vortrefflich demonstrierte, daß er sein
eigenes Pferd bis auf einen Strohhalm den Tag herunterbrachte, und ohne
Frage ein äußerst mutiges, kräftiges und gar nicht fressendes Tier
aus ihm gemacht haben würde, wenn es nicht vierundzwanzig Stunden vor
seinem ersten komfortablen vollkommenen Hungertage gestorben wäre. Die
mehrerwähnte Matrone wendete dasselbe System nicht selten mit gleichem
Unglücke auf die Kirchspielkinder an, deren nicht wenige vor Kälte oder
Hunger, oder weil sie einen Fall getan oder sich verbrannt hatten,
starben und zu ihren Vätern in jener Welt, die sie in dieser nicht
gekannt, versammelt wurden, wenn sie sie eben mit vieler Mühe so weit
gebracht hatte, daß sie von der möglichst geringen Quantität schwacher
Nahrungsmittel leben konnten.

Stellten die Direktoren unangenehme Untersuchungen über den Verbleib
eines Kindes an, oder taten die Geschworenen lästige Fragen, so
schützten das Zeugnis und die Aussage des Wundarztes und des
Kirchspieldieners gegen diese Zudringlichkeiten. Der erstere hatte
stets die Leichen geöffnet und nichts darin gefunden (was sehr
natürlich zuging), und der letztere beschwor stets, was dem Kirchspiel
angenehm war, und gab damit einen großen Beweis von Selbstaufopferung
und Hingebung. Das Armenkollegium besuchte von Zeit zu Zeit die
Filialanstalt und schickte tags zuvor den Kirchspieldiener, um seine
Ankunft zu verkünden. Und dann sahen die Kinder stets gut und reinlich
aus, und was konnte man mehr verlangen?

Es war nicht zu verlangen, daß die in der Filiale herrschende
Hausordnung ein allzu üppiges Gedeihen der Kinder beförderte, und
so war auch Oliver Twist an seinem neunten Geburtstage ein blasses,
schwächliches, im Wachstum zurückgebliebenes Kind von sehr geringem
Leibesumfange; doch wohnte in ihm ein gesunder, kräftiger Geist, der
auch, dank der strengen Diät des Hauses, hinreichenden Raum hatte, sich
auszudehnen. Oliver feierte seinen Geburtstag im Kohlenkeller in der
erlesenen Gesellschaft zweier anderer junger Herren, die nach einer
tüchtigen Tracht Schläge hier mit ihm eingesperrt worden waren, weil
sie sich erkühnt hatten, hungrig zu sein, als Frau Mann, die gutherzige
Pflegerin, durch die Erscheinung Mr. Bumbles, des Kirchspieldieners,
der dem Gartenpförtchen zuschritt, in Schrecken gesetzt wurde.

«Du meine Güte, sind Sie es, Mr. Bumble?» rief sie ihm aus dem Fenster,
anscheinend hoch erfreut, entgegen. -- «Susanne, bring gleich den
Oliver und die anderen beiden Buben herauf und wasch sie. Ach, Mr.
Bumble, wie lange haben Sie sich nicht sehen lassen!»

Mr. Bumble war ein wohlbeleibter und dazu cholerischer Mann, und so
rüttelte er, anstatt auf diese freundliche Begrüßung in höflicher Weise
zu antworten, wütend an der kleinen Pforte und gab ihr dann einen Stoß,
wie ihn nur ein Kirchspieldiener versetzen konnte.

«Herr des Himmels!» rief Mrs. Mann, indem sie aus dem Zimmer stürzte --
denn die drei Knaben waren inzwischen entfernt worden --, «daß ich es
auch dieser lieben Kinder wegen vergessen mußte, daß die Tür von innen
verriegelt ist. Treten Sie ein, Sir, bitte, treten Sie ein, Mr. Bumble!
Haben Sie die Güte.»

Obgleich diese Einladung von einem freundlichen Lächeln begleitet
war, das sogar das Herz eines Kirchenältesten erweicht haben würde,
besänftigte es den Kirchspieldiener doch keineswegs.

«Nennen Sie das einen respektvollen oder schicklichen Empfang, Mrs.
Mann,» fragte Bumble, indem er seinen Stab fester in die Hand nahm,
«wenn Sie die Kirchspielbeamten an Ihrer Gartenpforte warten lassen,
wenn sie in Parochialangelegenheiten in betreff der Parochialkinder
hierher kommen?»

«Ich kann Ihnen versichern, Mr. Bumble, daß ich nur ein paar der lieben
Kinder bei mir hatte, wegen deren Sie so freundlich sind, herzukommen»,
erwiderte Mrs. Mann mit großer Unterwürfigkeit.

Mr. Bumble hegte eine hohe Meinung von seiner oratorischen Begabung und
seiner Wichtigkeit. Er hatte die eine bewiesen und die andere gewahrt.
Er war in milderer Stimmung.

«Nun, nun, Mrs. Mann,» sagte er, «es mag sein, wie Sie sagen, es mag
sein. Lassen Sie mich hinein, Mrs. Mann; ich komme in Geschäften und
habe Ihnen etwas zu sagen.»

Mrs. Mann nötigte den Kirchspieldiener in ein kleines Sprechzimmer,
bot ihm einen Stuhl an und legte dienstbeflissen seinen dreieckigen
Hut und seinen Stab auf den Tisch vor ihm. Mr. Bumble wischte sich den
Schweiß von der Stirn, blickte freundlich auf den dreieckigen Hut und
lächelte. Ja, er lächelte. Kirchspieldiener sind auch nur Menschen, und
Mr. Bumble lächelte.

«Nehmen Sie es mir nicht übel, was ich Ihnen sagen will», bemerkte Mrs.
Mann mit bezaubernder Liebenswürdigkeit. «Sie wissen, Sie haben einen
weiten Weg hinter sich; wollen Sie nicht ein Gläschen nehmen?»

«Nicht einen Tropfen, nicht einen Tropfen», versetzte Mr. Bumble, indem
er mit seiner rechten Hand in würdevoller, aber freundlicher Weise
abwinkte.

«Ich denke, Sie werden mir schon den Gefallen tun», sagte Mrs. Mann,
die den Ton der Weigerung und die diese begleitende Gebärde bemerkt
hatte. «Nur ein ganz kleines Gläschen mit einem Schluck kalten Wassers
und einem Stück Zucker.»

Mr. Bumble hustete.

«Nur ein ganz kleines Gläschen», wiederholte Mrs. Mann in dringendem
Tone.

«Was ist es denn?» fragte der Kirchspieldiener.

«Nun, es ist das, von dem ich etwas im Hause zu halten verpflichtet
bin, um es den lieben Kindern in den Kaffee gießen zu können, wenn
sie nicht wohl sind, Mr. Bumble», entgegnete Mrs. Mann, während sie
ein Eckschränkchen öffnete und eine Flasche und ein Glas herausnahm.
«Es ist Genever, ich will Sie nicht hintergehen, Mr. Bumble. Es ist
Genever.»

«Geben Sie den Kindern Kaffee, Mrs. Mann?» fragte Bumble, der mit
seinen Augen den interessanten Vorgang der Mischung verfolgte.

«Ach, gesegne es ihnen Gott, ich tue es, so kostspielig es auch sein
mag», versetzte die Wärterin. «Ich könnte sie vor meinen leiblichen
Augen nicht leiden sehen, Sir, Sie wissen es ja.»

«Nein», sagte Mr. Bumble beistimmend; «nein, Sie könnten es nicht. Sie
sind eine menschlich denkende Frau, Mrs. Mann.» (Hier setzte sie das
Glas vor ihn hin.) «Ich werde so bald wie möglich Gelegenheit nehmen,
es dem Kollegium gegenüber zu erwähnen, Mrs. Mann.» (Er zog das Glas
näher zu sich heran.) «Sie empfinden wie eine Mutter.» (Er ergriff
das Glas.) «Ich -- ich trinke mit Vergnügen auf Ihre Gesundheit, Mrs.
Mann», und er trank es zur Hälfte aus.

«Und nun zu den Geschäften!» rief der Kirchspieldiener, indem er eine
lederne Brieftasche hervorzog. «Der Knabe, der halb auf den Namen
Oliver Twist getauft wurde, ist heute neun Jahre alt.»

«Des Himmels Segen über das liebe Herzchen!» rief Mrs. Mann aus und
mußte die Augen mit der Schürze abtrocknen.

Mr. Bumble fuhr fort: «Trotz ausgebotener Belohnung von zehn Pfund, ja
nachher von zwanzig Pfund -- trotz der übernatürlichen Anstrengungen
des Kirchspiels, sind wir nicht imstande gewesen, seinen Vater
ausfindig zu machen oder seiner Mutter Wohnung, Namen oder Stand in
Erfahrung zu bringen.»

«Wie geht es denn aber zu, daß er einen Namen hat?» fragte die
Waisenmutter.

Der Kirchspieldiener warf sich in die Brust und erwiderte: «Ich erfand
ihn.»

«Sie, Mr. Bumble!»

«Ich, Mrs. Mann. Wir benennen unsere Findlinge nach dem Alphabet. Der
letzte war ein S -- Swubble: ich benannte ihn. Dieser war ein T --
Twist: ich gab ihm abermals den Namen. Der nächste, der kommen wird,
wird Unwin heißen, der nächstfolgende Vilkins. Ich habe Namen im Vorrat
von A bis Z; und wenn ich beim Z angekommen bin, fang' ich beim A
wieder an.»

«Sie sind wirklich ein Gelehrter, Mr. Bumble!»

«Mag sein, mag sein, Mrs. Mann. Doch genug davon. Oliver ist jetzt
zu alt geworden zum Hierbleiben, das Kollegium hat beschlossen, ihn
zurückzunehmen, ich bin selbst gekommen, ihn abzuholen; -- wo ist er?»

Mrs. Mann eilte hinaus und erschien gleich darauf mit Oliver wieder,
der unterdes gewaschen und bestens gekleidet war.

«Mach 'nen Diener vor dem Herrn, Oliver», sagte sie.

Oliver verbeugte sich tief vor dem Kirchspieldiener auf dem Stuhle und
dem dreieckigen Hute auf dem Tische.

«Willst du mit mir gehen, Oliver?» redete ihn Mr. Bumble in feierlichem
Tone an.

Oliver war im Begriff, zu antworten, daß er auf das bereitwilligste mit
jedermann fortgehen würde, hob aber zufällig die Augen zu Mrs. Mann
empor, die hinter des Kirchspieldieners Stuhl getreten war und mit
grimmigen Mienen die Faust schüttelte. Er wußte nur zu gut, was das
bedeutete.

«Geht *sie* auch mit?» fragte er.

«Das ist unmöglich; sie wird aber bisweilen kommen und dich besuchen»,
erwiderte Bumble.

Das war kein großer Trost für Oliver; allein er hatte trotz seiner
Jugend Verstand genug, sich anzustellen, als verließe er das Haus nur
sehr ungern; ohnehin standen ihm die Tränen infolge des Hungers und
soeben erfahrener harter Züchtigung nahe genug. Mrs. Mann umarmte ihn
wiederholt und gab ihm, was er am meisten bedurfte, ein großes Stück
Butterbrot, damit er im Armenhause nicht zu hungrig anlangte. Die
Sache war natürlich abgemacht. Sein Butterbrot in der Hand, verließ
er die Stätte, wo kein Strahl eines freundlichen Blickes das Dunkel
seiner ersten Kinderjahre erhellt hatte. Und doch brach er in Tränen
kindlichen Schmerzes aus, als das Gartentor sich hinter ihm schloß.
Verließ er doch seine Leidensgefährten, die einzigen Freunde, die er in
seinem Leben gekannt hatte; und zum erstenmal seit dem Erwachen seines
Bewußtseins empfand er ein Gefühl seiner Verlassenheit in der großen,
weiten Welt. Mr. Bumble schritt kräftig vorwärts; der kleine Oliver
trabte neben ihm her und fragte am Ende jeder Meile, ob sie nicht bald
«da» sein würden. Auf diese Fragen gab Mr. Bumble sehr kurze und
mürrische Antworten; denn die zeitweilige Milde, die der Genuß von
Genever und Wasser in manchen Gemütern erzeugt, war längst verflogen,
und er war wiederum Kirchspieldiener.

Oliver war noch nicht eine Viertelstunde innerhalb der Mauern des
Armenhauses gewesen und hatte kaum ein zweites Stück Brot vertilgt,
als Mr. Bumble, der ihn der Obhut einer alten Frau übergeben hatte,
zurückkehrte. Er erklärte ihm, daß heute abend eine Sitzung des
Armenkollegiums stattfände, und daß er sofort vor diesem zu erscheinen
habe.

Oliver, der keine allzu klare Vorstellung von dem hatte, was ein
Armenkollegium zu bedeuten habe, war von dieser Mitteilung wie betäubt
und wußte nicht, ob er lachen oder weinen sollte. Er hatte jedoch keine
Zeit, über diesen Punkt nachzudenken; denn Mr. Bumble versetzte ihm mit
seinem Stabe einen Schlag auf den Kopf, um ihn aufzuwecken, und einen
anderen über den Rücken, um ihn munter zu machen. Dann befahl er ihm,
ihm zu folgen, und führte ihn in ein großes, weißgetünchtes Zimmer, in
dem acht bis zehn wohlbeleibte Herren um einen Tisch herumsaßen. Oben
am Tische saß in einem Armstuhl, der höher war als die übrigen, ein
besonders wohlgenährter Herr mit einem sehr runden, roten Gesichte.

«Mache dem Kollegium eine Verbeugung», sagte Bumble. Oliver zerdrückte
zwei oder drei Tränen in seinen Augen, und da er kein Kollegium,
sondern nur den Tisch sah, so machte er vor diesem eine wohlgelungene
Verbeugung.

«Wie heißt du, Junge?» begann der Herr auf dem großen Stuhle.

Oliver zitterte, denn der Anblick so vieler Herren brachte ihn gänzlich
außer Fassung; Bumble suchte ihn durch eine kräftige Berührung mit
dem Kirchspieldienerstabe zu beleben, und er fing an zu weinen. Er
antwortete daher leise und zögernd, worauf ihm ein Herr in weißer Weste
zurief, er wäre ein dummer Junge, was ein vortreffliches Mittel war,
ihm Mut einzuflößen.

«Junge,» sagte der Präsident, «höre, was ich dir sage. Du weißt doch,
daß du eine Waise bist?»

«Was ist denn das, Sir?» fragte der unglückliche Oliver.

«Er ist in der Tat ein dummer Junge -- ich sah es gleich», sagte der
Herr mit der weißen Weste sehr bestimmt.

«Du wirst doch wissen,» nahm der Herr wieder das Wort, der zuerst
gesprochen hatte, «daß du weder Vater noch Mutter hast und vom
Kirchspiel erzogen wirst?»

«Ja, Sir», antwortete Oliver, bitterlich weinend.

«Was heulst du?» fragte der Herr mit der weißen Weste; und es war in
der Tat höchst auffallend, daß Oliver weinte. Was konnte er denn für
eine Veranlassung dazu haben?

«Ich hoffe doch, daß du jeden Abend dein Gebet hersagst», fiel ein
anderer Herr in barschem Tone ein, «und für diejenigen, die dir
zu essen geben und für dich sorgen, betest, wie es sich für einen
Christenmenschen ziemt.»

«Ja, Sir», stotterte Oliver.

«Wir haben dich hierher bringen lassen,» sagte der Präsident, «damit du
erzogen werden und ein nützliches Geschäft lernen sollst. Du wirst also
morgen früh um sechs Uhr anfangen, Werg zu zupfen.»

Für die Vereinigung dieser beiden Wohltaten in der einfachen
Beschäftigung des Wergzupfens machte Oliver unter Nachhilfe des
Kirchspieldieners eine tiefe Verbeugung und ward dann eiligst in einen
großen Saal geführt, wo er sich auf einem rauhen, harten Bette in den
Schlaf weinte. Welch ein ehrenvolles Licht fällt hierdurch auf die
milden Gesetze Englands! Sie gestatten den Armen, zu schlafen!

Armer Oliver! Als er so in glücklicher Unbewußtheit seiner ganzen
Umgebung schlafend dalag, dachte er nicht daran, daß das Kollegium an
ebendemselben Tage zu einer Entscheidung gelangt war, die den größten
Einfluß auf seine künftigen Geschicke ausüben sollte. Die Sache
verhielt sich nämlich folgendermaßen: Die Mitglieder des Kollegiums
waren sehr weise, den Dingen auf den Grund gehende, philosophisch
gebildete Männer, und als sie dazu kamen, ihre Aufmerksamkeit dem
Armenhause zuzuwenden, fanden sie mit einem Male, was gewöhnliche
Sterbliche niemals entdeckt hätten. Den Armen gefiel es darin nur zu
gut! Es war ein regelrechter Unterschlupfsort für die ärmeren Klassen,
ein Gasthaus, in dem man nichts zu zahlen hatte -- ein Ort, an dem
man das ganze Jahr hindurch auf öffentliche Kosten das Frühstück, das
Mittagessen, den Tee und das Abendbrot einnehmen konnte -- ein Elysium
aus Ziegeln und Mörtel, in dem nur gescherzt und gespielt, aber nicht
gearbeitet wurde. «Oho,» sagte das Kollegium, «wir sind die richtigen
Männer, um hier Ordnung zu schaffen!» So ordneten sie denn an, daß alle
Armen die Wahl haben sollten (denn sie wollten um alles in der Welt
niemand zwingen), langsam in oder rasch außer dem Hause zu verhungern.
In dieser Absicht schlossen sie mit den Wasserwerken einen Vertrag
über die Lieferung einer unbegrenzten Menge Wasser und mit einem
Getreidehändler einen ebensolchen über die in großen Zwischenräumen
erfolgenden Lieferungen von kleinen Mengen Hafermehl ab und gaben
täglich drei Portionen eines dünnen Mehlbreies aus; außerdem wurde
zweimal wöchentlich eine Zwiebel und des Sonntags eine halbe Semmel
gereicht.

Die ersten sechs Monate nach der Aufnahme Oliver Twists war das System
in vollem Gange. Das Gemach, in welchem die Knaben gespeist wurden,
war eine Art Küche, und der Speisemeister, unterstützt von ein paar
Frauen, teilte ihnen aus einem kupfernen Kessel am unteren Ende ihre
Haferbreiportionen zu, einen Napf voll und nicht mehr, ausgenommen
an Sonn- und Feiertagen, wo sie auch noch ein nicht eben zu großes
Stück Brot bekamen. Die Näpfe brauchten nicht gewaschen zu werden,
denn sie wurden mit den Löffeln der Knaben so lange poliert, bis sie
wieder vollkommen blank waren; und auch an den Löffeln und Fingern
blieben Speisereste niemals hängen. Kinder pflegen eine vortreffliche
Eßlust zu besitzen. Oliver und seine Kameraden hatten drei Monate die
Hungerdiät ausgehalten, vermochten sie nun aber nicht länger mehr
zu ertragen. Ein für sein Alter sehr großer Knabe, dessen Vater ein
Garkoch gewesen, erklärte den übrigen, daß er, wenn er nicht täglich
zwei Näpfe Haferbrei bekomme, fürchten müsse, über kurz oder lang
seinen Bettkameraden, einen kleinen, schwächlichen Knaben, aufzuessen.
Seine Augen waren verstört, und rollten wild. Die halbverhungerte Schar
glaubte ihm, hielt einen Rat, loste darum, wer nach dem Abendessen zum
Speisemeister gehen und um mehr bitten solle, und das Los traf Oliver
Twist.

Der Abend kam, der Speisemeister stellte sich an den Kessel, der
Haferbrei wurde ausgefüllt und ein breites Gebet über der schmalen
Kost gesprochen. Die letztere war verschwunden, die Knaben flüsterten
untereinander, winkten Oliver, und die zunächst Sitzenden stießen
ihn an. Der Hunger ließ ihn alle Bedenklichkeiten und Rücksichten
vergessen. Er stand auf, trat mit Napf und Löffel vor den Speisemeister
hin und sagte, freilich mit ziemlichem Beben: «Bitt' um Vergebung, Sir,
ich möchte noch ein wenig.»

Der wohlgenährte, rotwangige Speisemeister erblaßte, starrte den
kleinen Rebellen wie betäubt vor Entsetzen an und mußte sich am Kessel
festhalten. Die Frauen waren vor Erstaunen, die Knaben vor Schreck
sprachlos. «Was willst du?» fragte der Speisemeister endlich mit
schwacher Stimme. Oliver wiederholte unter Furcht und Zittern seine
Worte, und nunmehr ermannte sich der Speisemeister, schlug ihn mit dem
Löffel auf den Kopf und rief laut nach dem Kirchspieldiener.

Das Armenkollegium war eben versammelt, als Mr. Bumble in großer
Erregung hereinstürzte und, zu dem Herrn auf dem hohen Stuhle gewandt,
sagte: «Mr. Limbkins, ich bitte um Verzeihung, Sir! Oliver Twist hat
mehr gefordert.»

Das Kollegium war starr. Entsetzen über eine solche Frechheit malte
sich auf allen Gesichtern.

«Mehr?» erwiderte Mr. Limbkins. «Fassen Sie sich, Bumble, und antworten
Sie mir klar und deutlich. Verstehe ich recht, daß er mehr gefordert
hat, nachdem er die von dem Direktorium festgesetzte Portion verzehrt
hatte?»

«Jawohl, Sir», entgegnete Bumble.

«Denken Sie an mich, Gentlemen,» sagte der Herr mit der weißen Weste,
«der Knabe wird dereinst gehängt werden.»

Niemand widersprach dieser Prophezeiung. Es entspann sich eine lebhafte
Diskussion. Oliver wurde auf Befehl des Kollegiums sofort eingesperrt,
und am nächsten Morgen wurde ein Anschlag an die Außenseite des Tores
geklebt, in dem jedermann, der Oliver Twist zu sich nehmen wollte, die
Summe von fünf Pfund zugesprochen wurde -- mit anderen Worten, man bot
Oliver Twist um fünf Pfund an jedermann aus, sei es Mann oder Frau,
der einen Lehrling oder Laufburschen brauchte, gleichviel wer und in
welchem Handwerke oder Geschäfte.




3. Kapitel.

    Berichtet, wie Oliver Twist nahe daran war, eine Anstellung zu
    bekommen, welche keine Sinekure gewesen wäre.


Wenn es Oliver darum zu tun gewesen wäre, die Prophezeiungen des Herrn
mit der weißen Weste selbst wahr zu machen, so hätte er zum wenigsten
Zeit genug dazu gehabt; denn er blieb acht Tage lang eingesperrt.
Allein, um sich im Gefängnis zu erhängen, fehlte ihm erstlich ein
Taschentuch -- denn Taschentücher waren als Luxusartikel verpönt --,
und zweitens war er noch zu sehr Kind. Er weinte daher nur den langen
Tag über, und wenn die lange, grausige Nacht kam, so deckte er seine
Händchen über seine Augen, um nicht in die Dunkelheit starren zu
müssen, kroch in einen Winkel und versuchte zu schlafen. Aber immer
und immer wieder fuhr er vor Angst und Entsetzen aus seinem unruhigen
Schlummer empor und drängte sich dichter und dichter an die Wand
heran, als wäre selbst ihre kalte, harte Fläche ein Schutz für ihn in
der Finsternis und Einsamkeit, die ihn rings umgaben.

Es war indes dafür gesorgt, daß es ihm an Leibesbewegung, Gesellschaft
und religiösem Trost nicht mangelte.

Was die Leibesübungen betrifft, so war es schönes, kaltes Wetter,
und er durfte seine Waschungen jeden Morgen unter der Pumpe in einem
gepflasterten Hofe vornehmen in der Gegenwart des Herrn Bumble, der
durch wiederholte Anwendung seines Stabes dafür sorgte, daß er sich
nicht erkältete, und daß eine prickelnde Empfindung seinen Körper
durchlief. Was die Gesellschaft betrifft, so wurde er jeden zweiten Tag
in den Saal geführt, wo die Knaben ihr Mittagbrot verzehrten, und wo er
vor deren Augen zum warnenden Beispiel ausgepeitscht wurde. Und weit
entfernt, daß ihm die Segnungen des religiösen Zuspruchs vorenthalten
worden wären, wurde er vielmehr jeden Abend zur Gebetsstunde in
denselben Raum gestoßen; hier durfte er zuhören und seinem Gemüte
Tröstung zuführen, da auf Anordnung des Kollegiums ein allgemeines
Gebet der Knaben eingefügt worden war, das eine besondere Klausel
enthielt, in der sie zu Gott flehten, er möge sie gut, tugendhaft,
zufrieden und gehorsam machen und vor der Sündhaftigkeit und
Lasterhaftigkeit Oliver Twists bewahren.

Während Olivers Angelegenheiten sich in diesem vielversprechenden und
günstigen Zustande befanden, ereignete es sich eines Morgens, daß der
Schornsteinfegermeister Mr. Gamfield auf der Landstraße langsam seines
Weges zog, in tiefem Sinnen über die Mittel und Wege, wie er seine
Miete, wegen deren er von seinem Hauswirt schon zu wiederholten Malen
gemahnt worden war, bezahlen sollte. Mr. Gamfield mochte den Stand
seiner Finanzen noch so sanguinisch betrachten: es fehlten ihm immer
noch fünf Pfund an der nötigen Summe, und in einer Art arithmetischer
Verzweiflung zermarterte er sein Gehirn und mißhandelte seinen Esel,
als er, am Armenhause angelangt, den Anschlag am Tore erblickte.

«Brrr!» sagte Mr. Gamfield zu dem Esel.

Der Esel war ebenfalls in tiefes Nachdenken versunken und beschäftigte
sich wahrscheinlich gelegentlich mit der Frage, ob er einen oder zwei
Kohlstrünke erhalten würde, wenn er die beiden Säcke Ruß, mit denen der
kleine Karren beladen war, an Ort und Stelle gebracht hätte, und so
trottete er denn weiter, ohne auf den Zuruf seines Herrn zu achten.

Mr. Gamfield stieß halblaut einen schweren Fluch aus, rannte dem Esel
nach und gab ihm einen Schlag auf den Kopf, der jeden anderen Schädel,
ausgenommen den eines Esels, zertrümmert haben würde. Dann ergriff er
den Zügel und riß scharf an dem Kinnbacken des Tieres, um ihm in zarter
Weise zu Gemüte zu führen, daß er nicht sein eigener Herr sei; durch
diese Mittel gelang es ihm, den Esel herumzulenken. Dann gab er ihm
einen zweiten Schlag auf den Kopf, um ihn bis zu seiner Rückkehr zu
betäuben, und schritt, nachdem er diese Vorsichtsmaßregeln getroffen
hatte, auf das Tor zu, um den Anschlag zu lesen.

Der Herr mit der weißen Weste stand, die Arme auf dem Rücken gekreuzt,
vor dem Tore, nachdem er in dem Beratungszimmer einige tiefempfundene
Wahrheiten zum besten gegeben hatte. Er hatte den kleinen Zwist
zwischen Mr. Gamfield und dem Esel beobachtet und lächelte höchst
vergnügt, als der Mann näher trat, um den Anschlag zu lesen, da er auf
den ersten Blick sah, daß Mr. Gamfield gerade der richtige Lehrherr für
Oliver sei. Auch Mr. Gamfield lächelte, als er das Schriftstück las,
denn fünf Pfund waren gerade die Summe, die er brauchte, und was den
Knaben betrifft, den er dazunehmen sollte, so wußte Mr. Gamfield, dem
es bekannt war, welcher Art die Kost im Armenhause war, daß es sich um
einen ganz kleinen, schmächtigen Kerl handeln würde, wie geschaffen für
die neuen Patentschornsteine. Daher las er den Anschlag noch einmal von
Anfang bis zu Ende durch, faßte als Beweis für seine Höflichkeit an
seine Pelzmütze und wandte sich an den Herrn in der weißen Weste.

«Dieser Junge hier, den das Armenhaus als Lehrling vergeben will ...»
begann Mr. Gamfield.

«Ach, lieber Mann,» erwiderte der Mann in der weißen Weste
herablassend, «was ist mit ihm?»

«Wenn das Kirchspiel ihn ein leichtes, angenehmes Handwerk, das
achtungswerte Schornsteinfegerhandwerk, erlernen lassen will, so
brauche ich einen Lehrling und bin bereit, ihn zu nehmen.»

«Treten Sie näher», entgegnete der Mann in der weißen Weste. Mr.
Gamfield lief erst noch einmal zurück, um dem Esel noch einen Schlag
vor den Kopf zu versetzen und am Zaume zu reißen, als Warnung, er
möge es sich nicht etwa einfallen lassen, in seiner Abwesenheit
durchzugehen, und folgte dann dem Herrn mit der weißen Weste in das
Zimmer, wo Oliver diesen zuerst gesehen hatte.

«Es ist ein schmutziges Gewerbe», erwiderte Mr. Limbkins, als Mr.
Gamfield seinen Wunsch abermals vorgebracht hatte.

«Es ist auch schon vorgekommen, daß Knaben in den Schornsteinen
erstickt sind», sagte ein anderer Herr.

«Das kam nur daher,» versetzte Gamfield, «daß man das Stroh naß machte,
ehe man es im Kamin anzündete, um die Jungen herunterzuholen; es gab
nur Rauch, aber keine Flamme. Rauch aber ist ganz unzweckmäßig, um
einen Jungen herunterzuholen, denn er veranlaßt ihn nur zum Schlafen,
und das eben ist es, was er will. Jungens sind widerspenstig und faul,
meine Herren, und ein gutes Feuer ist das beste Mittel, sie rasch zum
Herunterkommen zu bringen. Es ist auch ein ganz humanes Mittel, denn
wenn sie in der Esse steckengeblieben sind, so arbeiten sie, wenn sie
sich die Füße verbrennen, aus Leibeskräften, sich loszumachen.»

Der Herr in der weißen Weste schien sich über diese Erklärung höchlich
zu belustigen, aber seine Heiterkeit wurde durch einen strafenden
Blick, den ihm Mr. Limbkins zuwarf, sofort gedämpft. Die Direktoren
berieten nun ein paar Minuten miteinander, aber in so leisem Tone, daß
nur die Worte «Ersparnis» und «guten Eindruck bei der Abrechnung», die
mit großem Nachdruck mehrmals wiederholt wurden, hörbar waren. Endlich
hörte das Geflüster wieder auf, und Mr. Limbkins begann, nachdem die
Herren mit feierlicher Miene wieder ihre Plätze eingenommen hatten:
«Wir haben Ihren Vorschlag in Erwägung gezogen, können ihn aber nicht
annehmen.»

«Unter keinen Umständen», fiel der Herr in der weißen Weste ein.

«Ganz entschieden nicht», erklärten die übrigen Mitglieder des
Kollegiums.

Da auf Mr. Gamfield der leise Verdacht ruhte, daß schon drei bis vier
Knaben in seinem Geschäfte das Leben eingebüßt hatten, so kam ihm der
Gedanke, das Kollegium könnte vielleicht in einer ganz unbegreiflichen
Laune daran Anstoß genommen haben. Bei der Art ihrer Geschäftsführung
war dies zwar ganz unwahrscheinlich; da er aber keinen besonderen
Wunsch hegte, diesem Gerüchte neue Nahrung zuzuführen, so drehte er
seine Mütze in den Händen und entfernte sich langsam von dem Tische.

«So wollen Sie mir ihn also nicht überlassen, meine Herren?» fragte
Gamfield, an der Türe stehenbleibend.

«Nein», erwiderte Mr. Limbkins; «wenigstens sind wir der Meinung, Sie
müßten mit einer geringeren als der ausgesetzten Summe zufrieden sein,
da es doch ein gar zu schmutziges Gewerbe ist.»

Mr. Gamfields Gesicht strahlte, als er rasch an den Tisch zurückkehrte
und sagte: «Was wollen Sie geben, meine Herren? Seien Sie doch nicht zu
hart gegen einen armen Mann!»

«Ich sollte meinen, drei Pfund zehn Schilling wären übergenug», gab Mr.
Limbkins zur Antwort.

«Zehn Schilling zu viel», warf der Herr in der weißen Weste ein.

«Nun,» versetzte Gamfield, «sagen wir vier Pfund, meine Herren. Sagen
wir vier Pfund, und Sie sind ihn auf immer los.»

«Drei Pfund zehn Schilling», versetzte Mr. Limbkins fest.

«Wir wollen den Unterschied teilen, meine Herren, drei Pfund fünfzehn
Schilling.»

«Nicht einen Pfennig mehr», lautete die feste Entgegnung Mr. Limbkins'.

«Sie sind verdammt hart gegen mich, meine Herren», versetzte Gamfield
niedergeschlagen.

«Ach, Unsinn», erwiderte der Herr in der weißen Weste. «Es ist ein
gutes Geschäft, selbst wenn Sie gar nichts dazu bekommen. Nehmen Sie
ihn nur, guter Mann. Er ist gerade der richtige Junge für Sie. Er
braucht ab und zu den Stock; das wird ihm sehr gesund sein, und seine
Beköstigung braucht auch nicht sehr kostspielig zu werden, denn er ist
nicht sehr verwöhnt worden, seit er hier geboren wurde. Ha, ha, ha!»

Mr. Gamfield blickte scheu auf die Herren rund um den Tisch, und
da er auf den Gesichtern aller ein Schmunzeln bemerkte, lächelte
er ebenfalls. Der Handel wurde geschlossen, und Mr. Bumble erhielt
sofort den Befehl, Oliver Twist am Nachmittag dem Friedensrichter zur
amtlichen Bestätigung des Lehrvertrages vorzuführen.

Demgemäß wurde der kleine Oliver zu seinem maßlosen Erstaunen aus
seinem Kerker befreit und erhielt den Befehl, ein frisches Hemd
anzuziehen. Er hatte kaum diese ungewohnte gymnastische Übung
beendet, als Mr. Bumble ihm eigenhändig einen Napf Hafergrütze und
das sonntägliche Deputat Brot brachte. Bei diesem furchtbaren Anblick
begann Oliver bitterlich zu weinen, denn er dachte ganz natürlich
nicht anders, als daß ihn das Kollegium zu irgendeinem nützlichen
Zwecke schlachten lassen wolle, denn sonst hätte es wohl schwerlich
angefangen, ihn in dieser Weise fett zu machen.

«Heul dir die Augen nicht rot, Oliver, sondern iß und sei dankbar»,
sagte Mr. Bumble in würdevollem Tone. «Du sollst in die Lehre gegeben
werden.»

«In die Lehre?» fragte das Kind zitternd.

«Jawohl, Oliver,» erwiderte Mr. Bumble. «Die gütigen Herren, die ebenso
viele Eltern für dich sind, da du keine eigenen hast, wollen dich in
die Lehre geben, damit du im Leben auf deinen eigenen Füßen stehen
kannst, und wollen einen Mann aus dir machen, obgleich die Summe,
die das Kirchspiel dafür zu bezahlen hat, drei Pfund zehn Schilling
beträgt -- drei Pfund zehn Schilling, Oliver! siebzig Schilling --
einhundertundvierzig Sixpences! und all das für ein so ungeratenes
Waisenkind, das niemand leiden kann.»

Als Mr. Bumble in seiner Rede innehielt, um Atem zu schöpfen, rollten
die Tränen dem armen Kinde die Wangen hinunter, und es schluchzte
bitterlich.

«Nun, laß gut sein, Oliver», sagte Mr. Bumble etwas weniger würdevoll,
denn er war mit der Wirkung seiner Beredsamkeit zufrieden. «Wisch
dir die Augen mit den Ärmeln deiner Jacke und weine nicht in deine
Hafergrütze. Das ist Dummheit.» Das war es sicherlich, denn es befand
sich schon genügend Wasser darin.

Auf dem Wege zum Friedensrichter schärfte Bumble Oliver auf das
dringlichste ein, daß alles, was er zu tun hätte, darin bestände,
recht glücklich auszusehen, und wenn der alte Herr ihn frage, ob er in
die Lehre gehen wolle, zu antworten, er freue sich schon sehr darauf.
Oliver versprach, beiden Weisungen nachzukommen, um so mehr, als Mr.
Bumble ihm in einem freundlichen Hinweise androhte, es würde ihm sonst
sehr schlecht ergehen. An Ort und Stelle angelangt, wurde er in ein
kleines Zimmer eingeschlossen, und Mr. Bumble sagte ihm, er solle hier
bleiben, bis er wiederkäme und ihn abholte.

So blieb denn der Knabe mit klopfendem Herzen eine halbe Stunde
allein. Nach deren Verlauf steckte Bumble seinen bloßen, nicht mit dem
dreieckigen Hut geschmückten Kopf herein und sagte laut: «Nun, Oliver,
mein Kind, komme jetzt zu dem Herrn!»

Während Mr. Bumble dies sagte, warf er dem Knaben einen grimmigen,
drohenden Blick zu und fügte leise hinzu: «Erinnere dich an das, was
ich dir gesagt habe, infamer Bengel!»

Oliver starrte bei diesem verschiedenen Ton der Anrede Mr. Bumble
unschuldig in das Gesicht, aber dieser Herr führte ihn in das
anstoßende Zimmer, dessen Tür offen stand, und schnitt ihm dadurch jede
weitere Bemerkung ab. Es war ein geräumiges Zimmer mit einem großen
Fenster. Hinter einem Pulte saßen zwei alte Herren mit gepuderten
Perücken, von denen der eine eine Zeitung las, während der andere
mit Hilfe einer Schildpattbrille ein kleines vor ihm liegendes Stück
Pergament prüfte. Mr. Limbkins stand vor dem Pulte auf der einen Seite,
Mr. Gamfield mit teilweise gewaschenem Gesichte auf der anderen.

Der alte Herr mit der Brille schlief über dem Stück Pergament
allmählich ein, und es entstand eine kurze Pause, nachdem Oliver, von
Mr. Bumble geführt, sich vor das Pult hingestellt hatte.

«Dies ist der Knabe, Euer Edeln», sagte Mr. Bumble.

Der alte Herr, der die Zeitung las, erhob einen Augenblick den Kopf und
stieß den anderen alten Herrn an, worauf dieser erwachte.

«Ah, das ist also der Knabe?» fragte er.

«Ja, dies ist er, Euer Edeln», erwiderte Mr. Bumble. «Mache dem Herrn
Friedensrichter eine Verbeugung, mein Kind.»

Oliver gehorchte und machte sein schönstes Kompliment, das um so tiefer
ausfiel, da er noch nie Herren mit gepuderten Perücken gesehen hatte.

«Der Knabe wünscht also Schornsteinfeger zu werden?» sagte der
Friedensrichter.

«Mit Gewalt,» sagte Bumble, «will's mit Gewalt werden, Euer Edeln;
würde übermorgen wieder entlaufen, wenn wir ihn morgen in ein anderes
Geschäft gäben.»

Der Friedensrichter wendete sich zu dem Schornsteinfeger.

«Und Sie versprechen, ihn gut zu behandeln, ordentlich zu speisen, zu
kleiden, und was weiter dazu gehört?»

«Wenn ich's einmal gesagt habe, daß ich's will, so ist's auch meine
Meinung, daß ich's will», erwiderte Gamfield barsch.

«Ihre Rede ist eben nicht fein, mein Freund; doch Sie scheinen ein
ehrlicher, geradsinniger Mann zu sein», bemerkte der alte Herr und
richtete seine Brille auf den Meister, auf dessen häßlichem Gesicht
die Brutalität deutlich zu lesen stand. Aber der Friedensrichter war
halb blind und halb kindisch, und so konnte man füglicherweise nicht
verlangen, daß er das bemerke, was anderen auf den ersten Blick auffiel.

«Ich hoffe, ich bin es, Sir», erwiderte Mr. Gamfield grinsend.

«Ich hege daran nicht den mindesten Zweifel, mein Freund», erwiderte
der alte Herr, setzte seine Brille fester auf die Nase und suchte nach
dem Tintenfaß.

Es war der kritische Augenblick in Olivers Schicksal. Hätte das
Tintenfaß dort gestanden, wo es der alte Herr vermutete, so würde
er seine Feder eingetaucht und den Vertrag unterzeichnet haben, und
Oliver wäre dann auf der Stelle fortgeschleppt worden. Da es sich aber
unmittelbar vor seiner Nase befand, so folgte daraus mit Notwendigkeit,
daß er überall auf dem Pulte nach ihm suchte, ohne es zu finden, und
da er nun beim Suchen auch gerade vor sich hinblickte, so fiel sein
Auge auf das bleiche, verstörte Antlitz Oliver Twists, der trotz aller
ermahnenden Blicke und Püffe Bumbles das abstoßende Äußere seines
zukünftigen Lehrmeisters mit einem aus Grauen und Furcht gemischten
Ausdruck betrachtete.

Der alte Herr hielt inne, legte die Feder aus der Hand und blickte von
Oliver zu Mr. Limbkins hinüber, der mit unbefangener, heiterer Miene
eine Prise Schnupftabak zu nehmen versuchte.

«Mein liebes Kind!» sagte der alte Herr, sich über sein Pult lehnend.
Oliver fuhr beim Klang seiner Stimme zusammen. Dies läßt sich
entschuldigen, denn die Worte wurden in freundlichem Tone gesprochen,
und ungewohnte Laute erschrecken jeden. Er zitterte heftig und brach in
Tränen aus.

«Mein liebes Kind,» begann der alte Herr von neuem, «du siehst bleich
und geängstet aus. Was ist dir?»

«Treten Sie ein wenig von ihm weg», sagte der andere Beamte, das Papier
weglegend und sich mit einem Ausdrucke reger Teilnahme vorbeugend.

«Nun, mein Kind, sage uns, was dir ist. Habe keine Furcht!» Oliver fiel
auf die Knie, hob die gefalteten Hände empor und flehte schluchzend,
man möge ihn in das finstere Gemach zurückbringen, hungern lassen,
schlagen, ja totschlagen -- nur aber mit dem schrecklichen Manne nicht
fortschicken.

«Nun,» sagte Mr. Bumble, indem er seine Hände mit der eindrucksvollsten
Feierlichkeit erhob und seine Augen emporschlug, «von allen
hinterlistigen, niederträchtigen Waisenkindern, die ich je gesehen
habe, bist du der erbärmlichste Kerl, Oliver.»

«Halten Sie Ihren Mund, Kirchspieldiener», rief ihm der zweite alte
Herr zu, als Mr. Bumble seine Rede beendet hatte.

«Ich bitte Euer Edeln um Verzeihung», erwiderte Bumble, der nicht recht
gehört zu haben glaubte. «Haben Euer Edeln zu mir gesprochen?»

«Jawohl. Halten Sie Ihren Mund!»

Mr. Bumble war starr vor Entsetzen. Einem Kirchspieldiener zu befehlen,
den Mund zu halten! Das war ja wirklich eine Umwälzung aller sittlichen
Begriffe!

Der Friedensrichter blickte auf seinen Kollegen, der in bezeichnender
Weise nickte.

«Ich muß dem Vertrage die Bestätigung versagen», erklärte er dann, das
Pergament unwillig zur Seite schiebend.

«Ich hoffe,» stotterte Mr. Limbkins, «Sie werden nicht geneigt sein,
lediglich auf das Zeugnis eines Kindes der Meinung Raum zu geben, daß
das Verfahren des Direktoriums einem Tadel unterliege.»

«Ich bin als Friedensrichter nicht berufen, eine Meinung darüber
auszusprechen», entgegnete der alte Herr. «Nehmen Sie den Knaben wieder
mit sich und behandeln Sie ihn gut. Er scheint es zu bedürfen.»

Man hatte den Anschlag heruntergenommen, am folgenden Morgen wurde
jedoch Oliver abermals um fünf Pfund ausgeboten.




4. Kapitel.

    Oliver Twist, dem eine neue Stellung angeboten wird, tritt in das
    bürgerliche Leben ein.


Die Direktoren hatten Bumble befohlen, Erkundigungen einzuziehen, ob
nicht etwa ein Stromschiffer eines Knaben bedürfe, wie man denn die
jüngeren Söhne und ebenso die Waisenkinder gern zur See schickt, um
sich ihrer zu entledigen. Gerade als der Kirchspieldiener zurückkehrte,
trat Mr. Sowerberry aus dem Hause, der Leichenbestatter des
Kirchspiels, der es trotz seiner Beschäftigung doch nicht wenig liebte,
zu scherzen.

«Ich habe soeben das Maß zu den beiden gestern abend gestorbenen
Frauenzimmern genommen, Mr. Bumble», rief er ihm entgegen und bot ihm
zugleich seine Dose, ein artiges kleines Modell eines Patentsarges.

«Sie werden noch ein reicher Mann werden, Mr. Sowerberry», bemerkte
Bumble.

«Möcht's wünschen; aber die Direktoren zahlen nur gar zu geringe
Preise.»

«Ihre Särge sind auch gar zu klein, Mr. Sowerberry.»

«Größere tun auch nicht not, Mr. Bumble, bei der neuen Speiseordnung.»

Bumble mißfiel die Wendung, welche das Gespräch genommen; er suchte es
daher auf einen anderen Gegenstand zu lenken, spielte mit einem seiner
großen Rockknöpfe mit dem Kirchspielsiegelemblem -- dem barmherzigen
Samariter -- und begann von Oliver Twist zu sprechen.

«Beiläufig,» fing er an, «wissen Sie niemand, der einen Knaben braucht?
Einen Parochiallehrling, der gegenwärtig eine bloße Last, ein dem
Kirchspiel am Halse hängender Mühlstein, möchte ich sagen, ist. Sehr
günstige Bedingungen, Mr. Sowerberry, sehr günstige Bedingungen!»

Bei diesen Worten erhob Mr. Bumble seinen Stab zu dem Anschlage über
ihm und schlug dreimal auf die Worte «fünf Pfund», die mit riesengroßen
Buchstaben gedruckt waren. Dann fuhr er fort: «Nun, wie denken Sie
darüber?»

«Oh!» erwiderte der Leichenbestatter; «nun, Sie wissen, Mr. Bumble, ich
bezahle eine anständige Summe zu den Armenlasten.»

«Hm!» bemerkte Mr. Bumble. «Nun?»

«Nun,» antwortete der Leichenbestatter, «ich glaube, daß, wenn ich
so viel für die Armen bezahle, ich auch das Recht habe, so viel wie
möglich aus ihnen herauszuschlagen, Mr. Bumble, und so -- und so
beabsichtige ich denn, den Knaben selber zu nehmen.»

Mr. Bumble faßte den Leichenbestatter beim Arme und führte ihn in
das Haus. Mr. Sowerberry blieb fünf Minuten bei den Direktoren, und
es wurde abgemacht, daß Oliver noch am selbigen Abend «auf Probe» zu
ihm gehen sollte, was soviel sagen will, als daß der Meister, dem ein
Kirchspielknabe als Lehrling übergeben wird, denselben auf eine Anzahl
Lehrjahre haben soll, um mit ihm zu tun, was ihm beliebt, wenn er nach
kurzer Probezeit ersieht, daß ihm der Knabe genug arbeitet, ohne zu
eßlustig und also zu kostspielig zu sein. Dem kleinen Oliver wurde
gesagt, wenn er nicht gutwillig ginge oder sich im Armenhause wieder
blicken ließe, so würde man ihn nach gebührender Züchtigung zur See
schicken, wo er unfehlbar ertrinken müsse. Er zeigte wenig Rührung und
wurde nunmehr für gänzlich verhärtet erklärt. Er hatte freilich in
Wahrheit nicht zu wenig, sondern eher zu viel Gefühl, war aber durch
die erfahrene Behandlung betäubt und für den Augenblick vollkommen
abgestumpft. Auf dem Wege zu Mr. Sowerberry ermahnte ihn Bumble in
seinem gewöhnlichen Tone. Oliver traten die Tränen in die Augen.

«Was weinst du, Schlingel? Hab' ich's nicht immer gesagt, daß du die
schlechteste, undankbarste Kreatur von der Welt bist? Was hast du?
Sprich!»

«Ich bin so verlassen, Sir -- so ganz verlassen! Jedermann ist so
schlimm gegen mich. Es ist mir, als wenn ich hier blutete und mich
totbluten müßte»; -- und er preßte die Hand auf das Herz und blickte
mit nassen Augen seinem Führer in das Gesicht.

Bumble hustete, sagte endlich: «Trockne nur deine Augen und sei ein
guter Junge», und ging schweigend weiter.

Der Leichenbestatter, der soeben die Fensterladen seines Geschäfts
geschlossen hatte, machte gerade bei dem Scheine einer elenden Kerze
einige Eintragungen in sein Rechnungsbuch, als Mr. Bumble eintrat.

«Aha!» sagte er, von dem Buche aufblickend und mitten in einem Worte
aufhörend, «sind Sie es, Bumble?»

«Niemand anders!» erwiderte der Kirchspieldiener. «Hier ist er! Ich
habe Ihnen den Knaben gebracht.» Oliver machte eine Verbeugung.

«Ah, dies ist also der Knabe?» fragte der Leichenbestatter, indem er
die Kerze in die Höhe hob, um Oliver besser betrachten zu können.
«Liebe Frau,» rief er dann, «wolltest du vielleicht die Freundlichkeit
haben, einmal herzukommen?»

Mrs. Sowerberry tauchte aus einem kleinen Zimmer hinter dem Laden
auf und zeigte sich in der Gestalt einer kleinen, hageren Frau mit
zänkischem Gesichtsausdruck.

«Liebe Frau,» sagte der Leichenbestatter, «dies ist der Knabe aus dem
Armenhause, von dem ich dir erzählt habe.» Oliver machte abermals eine
Verbeugung.

«Mein Himmel, wie klein er ist!» rief Mrs. Sowerberry aus.

«Er ist allerdings klein», sagte Bumble, Oliver sehr unwillig
anblickend, als ob es des Knaben Schuld gewesen wäre, daß er nicht
größer war; «er wird aber größer werden, Mrs. Sowerberry.»

«O ja, auf unsere Kosten», entgegnete sie verdrießlich. «Ich sehe keine
Ersparnis mit Kirchspielkindern; sie kosten allezeit mehr, als sie wert
sind. Die Männer glauben aber immer, alles am besten zu wissen.»

Bei diesen Worten öffnete sie eine Seitentür und stieß Oliver eine
Treppe hinunter in ein finsteres, dumpfes Gelaß, den Vorraum des
Kohlenkellers und «Küche» genannt, und befahl einer schlumpigen
Dienstmagd, ihm zu geben, was für den nicht nach Hause gekommenen Trip
zurückgestellt wäre.

O daß doch so mancher, dessen Blut von Eis und dessen Herz von Stein
ist und der dennoch eine Stimme sich anmaßt, eine Stimme hat, wo es der
Beurteilung der Lage, dem Wohl oder Wehe der Armen gilt, den Knaben
hätte verschlingen sehen können, was der Haushund verschmäht! Wie sehr
wäre so vielen Menschenfreunden dieselbe und keine andere Diät zu
wünschen!

Frau Sowerberry hatte dem Knaben in stummem Entsetzen und mit trüben
Ahnungen in betreff seines künftigen Appetits zugeschaut; er hörte auf
zu essen, als er nichts mehr fand.

«Bist du endlich fertig?» sagte sie. «Nun komm, dein Bett ist unter
dem Ladentische. Du wirst dich doch nicht grauen, zwischen Särgen zu
schlafen? Aber wenn du auch nicht wolltest, du bekommst keine andere
Schlafstelle.»

Oliver folgte schüchtern und geduldig seiner neuen Herrin.




5. Kapitel.

    Oliver unter neuen Umgebungen und bei einem Leichenbegängnisse.


Sobald Oliver im Laden des Leichenbestatters allein gelassen
war, setzte er seine Lampe auf eine Bank, und Furcht und Grauen
durchschauerte ihn. Mitten im Gemach stand ein neuer, fast fertiger
Sarg; die schon zugeschnittenen, an die Wände umher gelehnten Bretter
erschienen ihm beim matten Lampenlichte wie Geister. Auf dem Boden
lagen große Nägel, Holzspäne, Stücke schwarzen Tuchs und Sargembleme,
und an der Wand über dem Ladentische hing das grauenhafte Bild eines
Leichenzuges. Die Luft war drückend heiß; sie deuchte Oliver wie
Grabesluft, die Öffnung zu seiner Ruhestätte unter dem Ladentische wie
ein gähnendes Grab.

Er fühlte sich allein und verlassen in der Welt, und obwohl er keinen
Schmerz über Trennung von Freunden oder Angehörigen empfand, so war ihm
das Herz dennoch schwer; und als er in sein enges Bett hineinkroch,
wünschte er, daß es sein Sarg sein und daß er darin hinaus auf den
Kirchhof getragen werden möchte, wo das hohe stille Gras über ihm
wüchse und im Winde säuselte und das Läuten der alten, traurigen
Turmglocke ihm schöne Träume zuführte in seinem süßen Schlummer.

Er wurde am folgenden Morgen durch ein lautes Pochen an der Ladentür
aus seinem unruhigen Schlafe geweckt; dasselbe wiederholte sich, ehe er
in seine Kleider schlüpfen konnte, ungefähr fünfundzwanzigmal und in
ungestümer Weise. Als er die Kette zu lösen begann, hörten die Beine zu
stoßen auf, und eine Stimme ließ sich vernehmen.

«Öffne die Tür, wird's bald?» rief die Stimme, die zu den Beinen
gehörte.

«Sofort, Sir!» erwiderte Oliver, indem er die Kette losmachte und den
Schlüssel umdrehte.

«Ich vermute, du bist der neue Lehrjunge, nicht wahr?» sprach die
Stimme durch das Schlüsselloch.

«Ja, Sir!» antwortete Oliver.

«Wie alt bist du?» fragte die Stimme weiter.

«Zehn Jahre, Sir!» entgegnete Oliver.

«Dann werde ich dich prügeln, wenn ich hineinkomme», sagte die Stimme;
«du wirst gleich sehen, daß ich es tue, du Armenhäusler!»

Oliver hatte schon zu oft das angedrohte Schicksal über sich ergehen
lassen müssen, um den leisesten Zweifel zu hegen, daß der Besitzer der
Stimme, wer es auch sein mochte, sein Versprechen wahr machen würde. Er
schob den Riegel mit zitternder Hand zurück und öffnete die Tür.

Ein paar Sekunden lang blickte Oliver die Straße auf und ab, weil er
glaubte, der unbekannte Besucher, der ihn durch das Schlüsselloch
angeredet hatte, habe sich einige Schritte entfernt, um sich zu
erwärmen; denn es war niemand zu sehen, außer einem großen Armenknaben,
der auf einem Pfosten vor dem Hause saß und ein Butterbrot verzehrte.

«Verzeihen Sie, Sir,» sagte Oliver endlich, da er keinen anderen
Besucher erblicken konnte, «haben Sie geklopft?»

«Ja, ich habe mit den Füßen an die Tür gestoßen», erwiderte der
Armenknabe.

«Wünschen Sie einen Sarg, Sir?» fragte Oliver unschuldig.

«Es wird nicht lange währen, bis du selbst einen brauchst,» war die
zornige Antwort, «wenn du Scherz mit Leuten treibst, die dir zu
befehlen haben. Weißt du nicht, wer ich bin? Noah Claypole, und du bist
mir untergeben, Musjö Ohnevater. Öffne die Fensterläden, Faulpelz!»

Oliver tat, wie ihm geheißen war, und gleich darauf erschien Mr. und
Mrs. Sowerberry. Oliver und sein neuer Tyrann wurden in die Küche
geschickt, um ihr Frühstück zu erhalten. Charlotte, die Köchin,
bedachte Noah gut und Oliver desto schlechter, der obendrein von jenem
sehr unsanft in einen dunklen Winkel gestoßen und vielfach gehänselt
wurde.

Noah war ein Freischüler, aber doch keine Waise aus dem Armenhause.
Sein Stammbaum war ihm sehr wohl bekannt; seine Eltern wohnten in der
Nachbarschaft. Seine Mutter war eine Waschfrau und sein Vater ein
pensionierter, täglich betrunkener Soldat. Die Ladenburschen nannten
ihn verächtlich «Lederhose» und so fort, was er schweigend duldete,
dagegen aber nunmehr mit desto größerem Übermut einen Schwächeren und
Elternlosen behandelte, den er als solchen tief unter sich sah. --
Welch ein köstlicher Stoff zu Betrachtungen über die liebenswürdige
menschliche Natur, deren vortreffliche Eigenschaften sich beim
hochstehenden Lord wie beim Armenknaben offenbaren!

Oliver hatte sich drei bis vier Wochen bei Mr. Sowerberry befunden,
als derselbe einst gegen seine Hausehre die Rede auf ihn brachte. «Der
Knabe sieht wirklich gut aus», bemerkte er.

«Kein Wunder,» entgegnete sie, «denn er ißt genug.»

«Er hat ein äußerst melancholisches Gesicht und sieht immer so
trübselig aus, daß er wirklich einen vortrefflichen Stummen[A] abgeben
würde.»

  [A] Die stummen Diener des Leichenbestatters, die vor den Türen der
  Trauerhäuser stehen.

Seine Gattin sah ihn verwundert an, und er fuhr fort: «Ich meine nicht
bei Erwachsenen, sondern bei Kinderbegräbnissen. 's ist etwas Neues,
auch zu dergleichen kleine Stumme zu stellen, und man kann sich etwas
davon versprechen.»

Mrs. Sowerberry, die für Geschäftssachen ein gutes Verständnis besaß,
war von der Neuheit des Gedankens überrascht; da es aber gegen ihre
Würde verstoßen haben würde, wenn sie dies zugegeben hätte, so fragte
sie nur mit großer Schärfe im Ton, warum ihr einfältiger Eheherr denn
nicht schon längst daran gedacht habe, und Mr. Sowerberry, der dies
richtig als Zustimmung auslegte, beschloß, Oliver in die Mysterien
des Leichenbestattergeschäftes einzuweihen und sich daher von ihm
zum ersten besten vorkommenden Begräbnisse begleiten zu lassen. Die
Gelegenheit ließ nicht lange auf sich warten, denn eine halbe Stunde
darauf erschien Bumble mit dem Auftrage zu einem Kirchspielbegräbnisse.

Mr. Sowerberry ordnete die erforderlichen Vorbereitungen an und befahl
Oliver, mit ihm zu gehen. Sie begaben sich nach dem bezeichneten Hause,
um das Maß zum Sarge zu nehmen, wo sich ihren Blicken eine Szene des
grauenvollsten Elends darbot, die auf Oliver, obgleich er an Elend so
wohl gewöhnt war, den peinlichsten Eindruck machte.

Am folgenden Tage, der rauh und regnerisch war, wiederholten sie
ihren Besuch, die Leiche wurde in den Sarg gelegt, jede Anordnung war
getroffen. Mr. Sowerberry sagte den Trägern, sie möchten sich sputen
und den Geistlichen nicht warten lassen; es wäre schon spät. Die Träger
setzten sich in eine Art von Trab, und Oliver mußte fast laufen, um
mitkommen zu können. Der Geistliche war noch nicht angelangt, der
Sarg wurde in einem entfernten Winkel des Kirchhofs neben der Gruft
einstweilen niedergesetzt, und Mr. Sowerberry und Bumble setzten sich
zum Küster in die Sakristei an das Feuer und nahmen die Zeitungen zur
Hand.

Nach einer halben Stunde erschien der Geistliche, Bumble verjagte die
Gassenbuben, die sich damit unterhielten, her- und hinüber über den
Sarg zu springen, der Geistliche las eilend die Gebete, entfernte sich
wieder, der Sarg wurde eingesenkt, die Grube zugeworfen, und alle
begaben sich auf den Heimweg.

«Nun, Oliver, wie hat dir's gefallen?» fragte Mr. Sowerberry.

«Recht gut, bedanke mich, Sir!» antwortete Oliver zögernd. «Aber doch
eigentlich nicht sehr gut.»

«Wirst dich schon daran gewöhnen», sagte der Leichenbesorger; «und 's
ist gar nichts, wenn du's erst gewohnt bist.»

Oliver hätte gern gewußt, wie lange es gedauert, ehe Mr. Sowerberry
sich daran gewöhnt, wagte jedoch nicht zu fragen und kehrte
gedankenvoll mit seinem Herrn nach Hause zurück.




6. Kapitel.

    In welchem Oliver kräftig auftritt.


Es trat gerade eine sehr ungesunde Zeit ein, und Oliver sammelte daher
in wenigen Wochen viel Erfahrung. Die Erfolge der scharfsinnigen
Spekulation Mr. Sowerberrys übertrafen alle seine Erwartungen. Die
ältesten Leute wußten sich nicht zu erinnern, daß so viele Kinder an
den Masern gestorben waren, und Oliver mit schwarzen, bis an die Knie
herunterreichenden Hutbändern führte einen Leichenzug nach dem andern
an. Die Mütter bewunderten ihn über die Maßen und waren unbeschreiblich
gerührt. Da er seinen Herrn auch zu den meisten Begräbnissen von
Erwachsenen begleiten mußte, um sich die für einen vollkommenen
Leichenbestatter so notwendige gemessene Ruhe und Selbstbeherrschung
anzueignen, so hatte er häufig Gelegenheit, die schöne Ergebung und
Seelenstärke zu bemerken, welche so viele Leute bei ihren schmerzlichen
Prüfungen und Verlusten beweisen.

Hatte Sowerberry zum Beispiel das Begräbnis einer reichen alten Dame
oder eines reichen alten Herrn zu besorgen, der von einer großen
Anzahl von Neffen und Nichten umgeben war, welche sich während seiner
Krankheit vollkommen untröstlich gezeigt und ihren Schmerz nicht einmal
vor den Augen des großen und größten Publikums hatten bemeistern
können, so blieb es selten aus, daß sie unter sich so heiter waren,
als man es nur wünschen konnte, und so froh und zufrieden miteinander
redeten oder auch lachten, als wenn sie ganz und gar keine Trübsal
erlebt hätten. Ehemänner ertrugen den Verlust ihrer Frauen mit der
heldenmütigsten Ruhe, und Ehefrauen legten die Trauerkleider um ihre
Männer auf eine Weise an, als wenn sie dadurch nicht etwa Schmerz
andeuten, sondern so anziehend als möglich erscheinen wollten. Viele
Damen und Herren, welche bei der Beerdigung der Verzweiflung nahe
zu sein schienen, beruhigten sich schon auf dem Heimwege und waren
vollkommen gefaßt, bevor die Teestunde vorüber war. Dieses alles war
sehr angenehm und lehrreich anzuschauen, und Oliver sah es mit großer
Bewunderung.

Daß das Beispiel so vieler Leidtragenden ihn zur Ergebung und Geduld
gestimmt hätte, kann ich mit Bestimmtheit nicht behaupten, sondern
vermag nur so viel zu sagen, daß er wochenlang mit Sanftmut die
Tyrannei und üble Behandlung ertrug, die er von seiten Noahs erfuhr,
der um so erbitterter gegen ihn wurde, weil sein Neid gegen ihn
erregt worden war. Charlotte mißhandelte ihn, weil es Noah tat, und
Mrs. Sowerberry war seine erklärte Feindin, weil ihr Gatte sich ihm
ziemlich freundlich erwies. Und so befand sich denn Oliver bei diesen
Feindschaften und fortwährender Leichenbegleitungslast nicht ganz so
behaglich wie das hungrige Ferklein, das aus Versehen in die Kornkammer
einer Brauerei eingeschlossen war.

Es muß aber jetzt ein an sich unbedeutender Vorfall erzählt werden, der
jedoch eine bedeutende Veränderung mit Oliver selbst wie mit seinen
Lebensschicksalen zur Folge hatte.

Sein Peiniger trieb seine gewöhnlichen Neckereien weiter als gewöhnlich
und hatte es offenbar darauf angelegt, ihn außer Fassung und zum Weinen
zu bringen, was ihm jedoch nicht gelingen wollte. Endlich sagte Noah
scherzend, er werde nicht verfehlen zuzuschauen, wenn Oliver gehängt
würde, und fügte hinzu: «Was wird aber deine Mutter dazu sagen -- und
wie geht's ihr denn?»

«Sie ist tot», entgegnete Oliver; «untersteh dich aber nicht, mir etwas
Schlechtes über sie zu sagen.»

Oliver wurde feuerrot, als er das sagte; er atmete rasch, um Mund
und Nase zuckte es ihm eigentümlich, und Claypole hielt dies für ein
untrügliches Anzeichen, daß Oliver bald heftig weinen werde. In dieser
Überzeugung ging er in seiner Quälerei weiter.

«Woran starb sie denn, Armenhäusler?» fragte er.

«An Kummer und Herzleid, wie mir eine unserer alten Wärterinnen gesagt
hat,» erwiderte Oliver, mehr, wie wenn er mit sich selbst redete, als
Noahs Frage beantwortend. «Ich glaube, daß ich's weiß, was es heißt,
daran zu sterben!»

Über seine Wange rollte eine Träne hinab, Noah pfiff eine muntere Weise
und sagte darauf: «Was hast du denn zu plärren -- um deine Mutter?»

«Daß du mir kein Wort mehr von ihr sagst -- sonst nimm dich in acht!»
rief Oliver.

«Ich soll mich in acht nehmen -- ich -- mich in acht nehmen vor einem
solchen unverschämten Tunichtgut? Und von wem soll ich kein Wort mehr
sagen? Von deiner Mutter? Die mag auch die rechte gewesen sein -- ha,
ha, ha!»

Oliver verbiß seine Pein und schwieg. Noah nahm den Ton spöttischen
Mitleids an.

«Nun, nun, sei nur ruhig; 's ist nichts mehr dran zu ändern, und ich
bedaure dich, wie's alle tun. Indes ist das wahr, ich weiß es, deine
Mutter taugte nichts; sie ist eine ganz verworfene Person gewesen.»

«Was sagst du?» rief Oliver rasch aufblickend.

«Eine ganz verworfene Person,» erwiderte Noah kühl, «und es war nur
gut, daß sie starb, denn es würde ihr jetzt schlecht genug ergehen in
der Tretmühle, wenn sie anders nicht deportiert oder gehängt worden
wäre. Hab' ich nicht recht, Armenhäusler?»

Olivers Geduld war zu Ende; purpurrot vor Wut sprang er auf, warf
seinen Stuhl samt dem Tische um, faßte Noah bei der Kehle, schüttelte
ihn so stark, daß ihm die Zähne im Munde klapperten, sammelte seine
ganze Kraft und schlug ihn mit einem einzigen Schlage zu Boden.

Eine Minute vorher hatte er das Aussehen des stillen, sanftmütigen,
eingeschüchterten Kindes noch gehabt, zu dem harte Behandlung ihn
gemacht hatte. Aber sein Mut war endlich erwacht; die tödliche
Beleidigung, die Noah seiner toten Mutter zugefügt, hatte sein Blut
in Wallung gebracht. Seine Brust hob sich, er stand aufrecht da wie
ein Held, sein Auge strahlte lebhaft; sein ganzes Wesen war verändert,
als er funkelnden Blickes vor dem feigen Quäler stand, der jetzt
zusammengekrümmt zu seinen Füßen lag.

«Er ermordet mich!» heulte Noah. «Charlotte, Fräulein! Der neue
Lehrjunge ermordet mich! Zu Hilfe, zu Hilfe! Oliver ist verrückt
geworden! Char--lotte!»

Noahs Geschrei wurde durch ein lautes Aufkreischen von Charlottes Seite
und durch ein lauteres von seiten Mrs. Sowerberrys beantwortet; die
erstere stürzte durch eine Seitentür in die Küche, während die letztere
noch auf der Treppe zauderte, bis sie sich völlig davon überzeugt
hatte, daß sie näher treten konnte, ohne ihr kostbares Leben zu
gefährden.

«Du verdammter Halunke!» schrie Charlotte und packte Oliver kräftig am
Arme. «Du undankbarer, mordgieriger, abscheulicher Schuft!» Und dabei
schlug sie unausgesetzt aus Leibeskräften auf Oliver ein.

Charlottes Faust gehörte nicht zu den leichtesten, und jetzt kam ihr
auch noch Mrs. Sowerberry zu Hilfe, die in die Küche stürzte und ihn
mit der einen Hand festhielt, während sie ihm mit der anderen das
Gesicht zerkratzte. Bei diesem günstigen Stande der Angelegenheit erhob
sich auch Noah vom Fußboden und griff ihn von hinten an.

Dieser dreifache Angriff war zu heftig, als daß er lange hätte dauern
können. Als sie alle drei ermüdet waren und nicht länger zerren und
schlagen konnten, schleppten sie Oliver in den Kehrichtkeller und
schlossen ihn hier ein. Nachdem dies glücklich vollbracht war, sank
Mrs. Sowerberry auf einen Stuhl und brach in Tränen aus.

«Um Gottes willen, sie stirbt!» rief Charlotte. «Ein Glas Wasser,
liebster Noah! Spute dich!»

«O Charlotte», sagte Mrs. Sowerberry stöhnend, «was für ein Glück, daß
wir nicht alle in unseren Betten ermordet worden sind!»

«Ja, Madam,» lautete die Antwort, «das ist in der Tat ein Glück von
Gott. Der arme Noah! Er war schon halb ermordet, als ich hineinkam.»

«Armer Junge!» sagte Mrs. Sowerberry, indem sie mitleidig auf den
Knaben blickte. «Was sollen wir anfangen?» fuhr sie nach einer Weile
fort. «Der Herr ist nicht daheim; es ist kein Mann im ganzen Hause, und
er wird die Kellertür in zehn Minuten eingestoßen haben.»

«Mein Gott, mein Gott!» jammerte Charlotte, «ich weiß es nicht, Ma'am!
Aber vielleicht schicken wir nach der Polizei.»

«Oder nach dem Militär!» warf Claypole ein.

«Nein, nein!» erwiderte Mrs. Sowerberry, die sich in diesem Augenblick
an Olivers alten Freund erinnerte. «Lauf zu Mr. Bumble, Noah, und bitte
ihn, unverzüglich herzukommen und keine Minute zu verlieren. Es tut
nichts, wenn du auch ohne Mütze gehst. Mach hurtig!»

Ohne sich die Zeit zu einer Antwort zu lassen, stürzte Noah davon,
und die ihm begegnenden Leute waren sehr erstaunt, einen Armenknaben
barhäuptig in voller Eile durch die Straßen rennen zu sehen.




7. Kapitel.

    Oliver bleibt widerspenstig.


Noah Claypole unterbrach seinen hastigen Lauf nicht ein einziges Mal
und kam ganz atemlos vor dem Tor des Armenhauses an. Hier blieb er
einen Augenblick stehen, um sein Gesicht in möglichst klägliche Falten
zu legen, klopfte dann laut an die Pforte und zeigte dem öffnenden
Armenhäusling eine so jammervolle Miene, daß selbst dieser, der sein
ganzes Leben lang nichts als jammervolle Mienen um sich gesehen hatte,
erschrocken zurückfuhr und fragte: «Was hast du denn nur, Junge?»

«Mr. Bumble, Mr. Bumble!» rief Noah in gut geheuchelter Angst und in so
lautem, erregtem Tone, daß Mr. Bumble, der zufällig in der Nähe war,
es nicht nur hörte, sondern auch dadurch in solche Aufregung geriet,
daß er ohne seinen dreieckigen Hut in den Hof stürzte -- ein deutlicher
Beweis dafür, daß selbst ein Kirchspieldiener unter Umständen seine
Fassung verlieren und seine persönliche Würde außer acht lassen kann.

«Oh, Mr. Bumble -- o Sir!» schrie Noah; «Oliver, Sir -- Oliver Twist!»

«Wie -- was? Ist er -- ist er davongelaufen?»

«Nein, Sir; er ist ganz ruchlos geworden. Er hat mich und Charlotte und
Missis ermorden wollen! O Sir! o Sir -- mein Nacken, mein Kopf, mein
Leib, mein Leib!»

Sein Geheul zog den Herrn mit der weißen Weste herbei.

«Sir,» rief Bumble demselben entgegen, «hier ist ein Knabe aus der
Freischule, der von Oliver Twist beinahe ermordet worden wäre!»

«Bei Gott,» bemerkte der Herr mit der weißen Weste, «das habe ich
gewußt. Ich hatte von Anfang an eine seltsame Ahnung, daß dieser
freche, kleine Taugenichts noch gehängt werden würde.»

«Er hat auch die Magd ermorden wollen», sagte Bumble mit bleichem
Gesicht.

«Und die Frau», fiel Noah ein.

«Und nicht wahr, Noah, sagtest du nicht, auch seinen Herrn?» fragte
Bumble.

«Nein, der Herr war nicht zu Hause, sonst hätte er ihn auch gemordet»,
antwortete Noah. «Aber der Bösewicht sagte, er wollte es tun.»

«Sagte er, daß er es tun wollte, mein Kind?» fragte der Herr mit der
weißen Weste.

«Ja, Sir!» erwiderte Noah. «Und Missis wünscht zu wissen, ob Mr.
Bumble wohl nicht einen Augenblick Zeit hätte, um zu kommen und ihn zu
züchtigen, da der Herr nicht zu Hause ist.»

«Gewiß, mein Junge, gewiß», sagte der Herr in der weißen Weste, indem
er freundlich lächelte und Noahs Kopf streichelte. «Du bist ein guter
Junge, ein sehr guter Junge. Hier hast du einen Penny. Bumble, gehen
Sie sofort mit Ihrem Stabe zu Sowerberry und sehen Sie zu, was am
besten zu tun ist. Schonen Sie ihn nicht, Bumble, und sagen Sie auch
Sowerberry, er solle in Zukunft strenge mit ihm verfahren.»

«Ich werde alles zu Ihrer vollen Zufriedenheit besorgen, Sir!»
erwiderte Bumble, indem er sich zusammen mit Noah auf den Weg machte.

Als sie an ihrem Bestimmungsorte anlangten, war die Lage der Dinge
dort unverändert. Sowerberry war noch nicht zurückgekehrt, und Oliver
schlug fortwährend mit unverminderter Heftigkeit an die Kellertür. Mr.
Bumble donnerte mit seinem Fuße von außen an die Tür, um sein Kommen
anzuzeigen, legte dann seinen Mund ans Schlüsselloch und sagte in
tiefem, eindringlichem Tone: «Oliver.»

«Laßt mich hinaus!» rief Oliver von innen.

«Kennst du meine Stimme, Oliver?»

«Ja!»

«Fürchtest du dich nicht -- zitterst du nicht bei meiner Nähe?»

«Nein!»

Bumble war starr vor Erstaunen.

«Er muß verrückt geworden sein!» bemerkte Mrs. Sowerberry.

«'s ist keine Verrücktheit, Ma'am,» sagte Bumble, «'s ist das Fleisch!»

«Das Fleisch?!»

«Ja, ja, Ma'am! Sie haben ihn überfüttert, Ma'am. Hätten Sie ihm nichts
als Haferbrei gegeben, so würde er nimmermehr so geworden sein.»

Mrs. Sowerberry machte sich wegen ihrer Gutherzigkeit und Freigebigkeit
die bittersten Vorwürfe, so unschuldig in Gedanken, Worten und Werken
sie auch war.

Bumble erklärte, daß nur Einsperren und sodann strenge Diät den
rebellischen Sinn des kleinen Galgenstricks würden bändigen können.
In diesem Augenblick kehrte Sowerberry zurück, dem sofort der Vorfall
mit solchen Übertreibungen erzählt wurde, daß er die Tür öffnete, den
Knaben beim Kragen faßte und herauszog.

Olivers Kleider waren zerrissen, sein Gesicht war verschwollen und
zerkratzt, und sein Haar hing ihm wirr über die Stirn herab. Die
zornige Röte war jedoch aus seinem Gesicht nicht verschwunden, und als
er aus seinem Gefängnis gezogen wurde, warf er Noah einen drohenden
Blick zu.

«Nun, du bist ja ein netter Bursche», sagte Sowerberry, schüttelte
Oliver derb und gab ihm rechts und links ein paar Ohrfeigen.

«Er beschimpfte meine Mutter», sagte Oliver.

«Und wenn er das auch tat, du undankbarer Bösewicht», versetzte Mrs.
Sowerberry. «Sie hat's verdient, was er von ihr gesagt hat, und noch
viel mehr.»

«Nein, nein!» rief Oliver. «'s ist eine Lüge!»

Mrs. Sowerberry brach in eine Tränenflut aus, und dies ließ ihrem
Gatten keine Wahl. Denn wenn er nicht auf der Stelle Oliver
nachdrücklich gezüchtigt hätte, so würde er sich, gemäß allen
Ehezänkereiregeln, als eine Nachtmütze, ein liebloser Ehemann, ein
Ungeheuer gezeigt haben. So ungern er es daher auch tun mochte, er
züchtigte Oliver dermaßen, daß die nachträgliche Anwendung des Rohrs
Mr. Bumbles jedenfalls sehr unnötig war. Oliver wurde darauf bei Wasser
und Brot wieder eingesperrt und spät abends unter Noahs unbarmherzigem
Gespött zu Bett gewiesen.

Erst hier ließ er seinen Gefühlen freien Lauf. Er hatte allen Spott und
Hohn mit hartnäckiger Verachtung, die schmerzlichsten Streiche ohne
Schrei ertragen und würde nicht geweint haben, wenn man ihn lebendig
geröstet hätte; ein solcher Stolz war in seiner Brust erwacht. Nun
aber, da er allein und gänzlich sich selber überlassen war, fiel er auf
die Knie nieder, bedeckte das Gesicht mit den Händen und weinte solche
Tränen, wie Gott sie den Betrübten und Geängsteten zur Erleichterung
ihres Herzens sendet, wie nur wenige menschliche Wesen, so jung an
Jahren wie Oliver, sie zu vergießen Ursache hatten.

Es währte lange, bevor er sich wieder erhob. Das Licht war tief
heruntergebrannt, er horchte und blickte vorsichtig umher, öffnete
leise die Tür und sah hinaus. Die Nacht war finster und kalt. Die
Sterne schienen ihm weiter von der Erde entfernt zu sein, als er sie je
gesehen; die Bäume, von keinem Winde bewegt, standen wie Geister da. Er
verschloß die Tür wieder, knüpfte seine wenigen Habseligkeiten in ein
Taschentuch und setzte sich auf eine Bank, um den Anbruch des Tages zu
erwarten.

Mit dem ersten durch die Ritzen der Fensterladen eindringenden
Lichtstrahle stand er auf, öffnete die Tür zum zweiten Male, blickte
furchtsam umher, zögerte ein paar Augenblicke, trat hinaus und ging,
ungewiß, wohin er sich wenden sollte, rasch vorwärts. Nach einiger
Zeit gewahrte er, daß er sich ganz in der Nähe der Anstalt befände, in
der er seine ersten Kinderjahre verlebt hatte. Es war niemand zu hören
oder zu sehen; er blickte in den Garten hinein. Einer seiner kleinen,
weit jüngeren Spielkameraden reinigte ein Beet vom Unkraut. Sie hatten
miteinander gar oft Hunger, Schläge und Einsperrung erduldet.

«Pst! Dick!» rief Oliver.

Der Knabe lief herbei und streckte ihm die abgemagerten Hände durch die
Gittertür entgegen.

«Ist schon jemand auf, Dick?»

«Keiner als ich.»

«Sag' ja nicht, daß du mich gesehen hast, Dick; ich bin fortgelaufen;
konnt's nicht mehr aushalten und will mein Glück in der Welt versuchen.
Ich muß weit fort von hier; weiß nicht, wohin. Wie blaß du aussiehst!»

«Ich habe den Doktor sagen hören, daß ich sterben müßte. Ach, das ist
schön, daß du hier bist! Aber halt dich nicht auf; lauf fort!»

«Ja, ja, leb wohl! Ich weiß gewiß, wir sehen uns wieder, Dick. Du wirst
noch recht glücklich werden.»

«Das hoff' ich -- wenn ich tot bin; eher nicht. Ich weiß es, Oliver,
der Doktor hat recht; denn ich träume so viel vom Himmel und von Engeln
und freundlichen Gesichtern, die ich niemals sehe, wenn ich aufwache.
Leb wohl, Oliver; geh mit Gott! Gottes Segen begleite dich!»

Oliver hatte noch nie des Himmels Segen auf sich herabrufen hören, und
nie vergaß er diese Segnung von den Lippen eines Kindes unter allen
Leiden, Sorgen, Mühen, Kämpfen und Wechselschicksalen seines Lebens.




8. Kapitel.

    Oliver geht nach London und trifft mit einem absonderlichen jungen
    Gentleman zusammen.


Oliver lief ohne Rast und Ruhe, bis er um die Mittagsstunde bei einem
Meilensteine stillstand, auf dem die Entfernung Londons angegeben
war. Dort konnte man ihn nicht finden, er hatte oft sagen hören, daß
die unermeßliche Stadt zahllose Mittel zum Fortkommen darböte, sein
Entschluß war gefaßt; er machte sich bald wieder auf den Weg und
gedachte nun erst der Schwierigkeiten, die er zu überwinden haben
würde, um an sein Ziel zu gelangen. Er hatte ein grobes Hemd, zwei
Paar Strümpfe, eine Brotrinde und einen Penny in seinem Bündel -- ein
Geschenk Mr. Sowerberrys nach einem Begräbnisse, bei welchem er sich
dessen ungewöhnliche Zufriedenheit verdient hatte. Er sann vergeblich
darüber nach, wie er mit so geringen Mitteln London erreichen solle --
und trabte weiter.

Nachdem er zwanzig Meilen zurückgelegt hatte, lenkte er auf eine Wiese
ein und legte sich in einem Heuhaufen zur Ruhe nieder. Er machte am
zweiten Tage abermals zwölf Meilen, verwendete seinen Penny für Brot,
übernachtete auf ähnliche Weise und erhob sich am dritten Morgen fast
erfroren und mit erstarrten Gliedern, so daß er sich kaum von der
Stelle bewegen konnte.

Die Straße wand sich hier einen ziemlich steilen Hügel hinauf, und er
flehte die Außenpassagiere einer Postkutsche um eine Gabe an. Nur einer
beachtete ihn, rief ihm zu, er möge warten, bis man oben angelangt
wäre, und begehrte darauf zu erfahren, wie weit er um einen halben
Penny mitlaufen könne. Oliver mußte nach der größten Anstrengung doch
bald zurückbleiben, und der Mildtätige steckte sein Geldstück wieder
in die Tasche und erklärte ihn für einen faulen Schlingel, der keine
Freigebigkeit verdiene. Dahin rollte die Postkutsche und ließ nur eine
Staubwolke zurück.

In manchen Dörfern waren Pfosten mit Tafeln errichtet, auf welchen
scharfe Drohungen gegen alle Bettler zu lesen waren, und Oliver eilte
furchtsam weiter; in anderen, wenn er etwa vor einem Gasthause mit
sehnsüchtigen Blicken stillstand, hieß man ihn sich davonmachen, wenn
er nicht als ein Dieb eingesperrt werden wollte. Aus vielen Häusern
vertrieb ihn die Drohung, daß man die Hunde loslassen werde, wenn er
sich nicht sofort entferne.

Es würde ihm ohne Zweifel ergangen sein, wie seiner unglücklichen
Mutter, wenn sich nicht ein menschenfreundlicher Schlagbaumwärter und
eine gutherzige Frau seiner angenommen hätten. Jener erquickte ihn
durch ein, wenn auch nur aus Brot und Käse bestehendes Mittagsmahl;
und diese, die einen schiffbrüchigen, sie wußte nicht wo umherirrenden
Großsohn hatte, gab ihm, was ihre Armut vermochte, und obendrein,
was mehr war für Oliver und ihn alle seine Leiden auf eine Zeitlang
vergessen ließ, freundliche Worte und mitleidige Zähren.

Am siebenten Morgen nach Sonnenaufgang erreichte er mit wunden Füßen
die kleine Stadt Varnet. Die Fensterläden waren geschlossen, die
Straßen waren leer; nicht eine einzige Seele hatte sich schon zu den
Geschäften des Tages erhoben. Die Sonne ging in all ihrer strahlenden
Schönheit auf; aber ihr Licht diente nur dazu, dem Knaben seine
Verlassenheit so recht zu Gemüte zu führen, als er mit blutenden Füßen
und staubbedeckt auf einer Türschwelle saß.

Allmählich wurden die Läden geöffnet und die Rouleaus in die Höhe
gezogen, und die Leute begannen auf und ab zu gehen. Einige blieben
stehen, um Oliver ein paar Augenblicke zu betrachten, oder wandten
sich im Vorbeieilen um, um einen Blick auf ihn zu werfen; aber niemand
kümmerte sich um ihn oder fragte, wie er dorthin käme. Er hatte nicht
den Mut, jemand um eine Gabe anzusprechen. Nach einiger Zeit ging ein
Knabe an ihm vorüber, sah sich nach ihm um, ging weiter, sah sich noch
einmal um, stand still, kehrte zurück und redete ihn an.

Er mochte ungefähr so alt sein wie Oliver selbst, der nie einen so
absonderlichen Kauz gesehen. Er hatte eine Stumpfnase und eine
platte Stirn, sah höchst ordinär und schmutzig aus, und seine ganze
Haltung und sein Benehmen war wie das eines Mannes. Er war klein für
sein Alter, hatte Dachsbeine und kleine, scharfe, häßliche Augen.
Der Hut saß ihm so lose auf dem Kopfe, als wenn er jeden Augenblick
herunterfallen müßte, und er würde auch heruntergefallen sein, wenn er
nicht durch häufige rasche Kopfbewegungen seines Besitzers immer wieder
zurechtgerückt oder befestigt worden wäre. Die Kleidung des Kleinen war
gleichfalls nichts weniger als knabenhaft, und die ganze Figur stellte
das vollkommene Bild eines renommierenden, prahlhaften kleinen Helden
von vier Fuß Höhe dar.

«Was fehlt dir, Bursch? Was scheft dermehr?»[B] redete er Oliver an.

  [B] Was gibt's?

«Ich bin sehr hungrig und müde», erwiderte Oliver, mit Tränen in den
Augen. «Ich komme weit her und bin seit sieben Tagen auf der Wanderung
gewesen.»

«Weit her -- hm! -- seit sieben Tagen auf der Wanderung gewesen? --
Ah -- sehe schon -- auf Oberschenkels Befehl -- he? Doch,» fügte er
hinzu, als er Olivers verwunderte Miene gewahrte, «du scheinst nicht zu
wissen, was ä Oberschenkel ist, mein guter Kochemer[C].»

  [C] Spitzbubenkamerad.

Oliver erwiderte schüchtern, er wisse allerdings sehr wohl, daß man
unter einem Oberschenkel den oberen Teil eines Beines verstehe.

«Ha, ha, ha! Wie grün!» rief der junge Gentleman aus. «Ä Oberschenkel
ist ä Friedensrichter, wer auf 'nes Oberschenkels Befehl geht, kommt
nicht vorwärts, sondern geht immer 'nauf, ohne wieder 'runter zu
kommen. Noch nicht in der Mühle gewesen?»

«In was für einer Mühle?» fragte Oliver.

«Ei, in der, die in ä Doves[D] Platz hat. Doch du bist butterich[E];
ich hab' freilich auch nicht eben zu viel Massumme[F], aber so weit's
zureicht, will ich rausrücken und blechen. Steh auf -- komm!»

  [D] Gefängnis.

  [E] Hungrig.

  [F] Geld.

Der junge Gentleman half Oliver aufstehen und nahm ihn mit sich in sein
Gasthaus, wo er Brot und Schinken bringen ließ und ihn sehr aufmerksam
beim Essen beobachtete. Als sich Oliver endlich gesättigt, warf er die
Frage hin: «Nach London?»

«Ja.»

«Hast du eine Wohnung?»

«Nein.»

«Geld?»

«Nein.»

Der junge Herr senkte die Hände in die Taschen und pfiff. --

«Wohnst du in London?» fragte Oliver.

«Ja, wenn ich zu Hause bin. Aber du weißt wohl nicht, wo du kommende
Nacht schlafen sollst?»

«Nein», antwortete Oliver. «Ich habe seit sieben Nächten unter keinem
Dache geschlafen.»

«Mach dir darum nur keine Sorgen. Ich gehe heute abend nach London und
kenne da 'nen respektablen alten Herrn, der dir Wohnung umsonst geben
und dir bald 'ne gute Stelle verschaffen wird -- das heißt, wenn dich ä
Schentleman einführt, den er kennt. Und ob er mich wohl kennt!» fügte
der junge Herr lächelnd hinzu.

Das unerwartete Anerbieten war zu lockend, als daß Oliver einen
Augenblick hätte anstehen sollen, es anzunehmen. Er wurde zutraulicher
und erfuhr nun auch, daß sein neuer Freund Jack Dawkins heiße und ein
besonderer Liebling des erwähnten alten Herrn sei. -- Jacks Äußeres
schien freilich den Lieblingen des alten Herrn nicht viele Vorteile zu
versprechen; allein da er ziemlich leichtfertig und großsprecherisch
redete und auch gestand, daß er unter seinen Bekannten allgemein den
Namen des «gepfefferten Baldowerers» (d. h. gewitzten Kundschafters)
führe, so schloß Oliver, er möge nicht eben viel taugen und die guten
Lehren seines Wohltäters in den Wind schlagen. Oliver nahm sich daher
in der Stille vor, sich so bald wie möglich die Gunst des alten Herrn
zu erwerben, und wenn er den Baldowerer unverbesserlich fände, die
Ehre der näheren Bekanntschaft mit ihm abzulehnen.

Da es Jack nicht genehm war, vor Abend in London einzutreffen, so wurde
es fast elf Uhr, bevor sie den Schlagbaum von Islington erreichten.
Der Baldowerer führte Oliver eiligen Schrittes durch ein Gewirr von
Straßen und Gassen, so daß sein Begleiter ihm kaum zu folgen vermochte.
Trotz dieser Eile konnte Oliver nicht umhin, beim Weitergehen ein paar
hastige Blicke nach beiden Seiten zu werfen. Eine schmutzigere oder
elendere Gegend hatte er noch nie gesehen. Die Straßen waren äußerst
eng und unsauber, und die Luft war mit üblen Gerüchen erfüllt. Es war
eine große Menge kleiner Läden vorhanden, aber der einzige Warenvorrat
schien in Haufen von Kindern zu bestehen, die selbst zu dieser späten
Nachtstunde innerhalb und außerhalb der Türen umherkrochen oder im
Innern der Häuser schrien. Bedeckte Wege und Höfe, die hier und da von
der Hauptstraße abbogen, führten zu kleinen Häusergruppen, vor denen
betrunkene Männer und Frauen sich tatsächlich im Schmutze wälzten,
und an verschiedenen Torwegen tauchten großgewachsene, verdächtig
aussehende Burschen auf, die allem Anschein nach nicht viel Gutes im
Schilde führten. Oliver überlegte schon, ob er nicht am besten täte,
davonzulaufen, als ihn sein Führer plötzlich beim Arm nahm, die Tür
eines Hauses unweit Fieldlane öffnete, ihn hineinzog und die Tür wieder
verschloß. Der Baldowerer pfiff und erwiderte auf den Ruf: «Wer da?»
-- «Grim und petacht!»[G] Unten auf dem Hausflur zeigte sich Licht,
und der Kopf eines Mannes tauchte auf der zur Küche hinunterführenden
Treppe empor.

  [G] Gut und sicher.

«Es sind euer zwei -- wer ist der andere?»

«Ein neuer Chawwer», rief Jack, Oliver nachziehend, zurück.

«Woher kommt er?»

«Von Grünland. Ist Fagin oben?»

«Ja. Er sortiert die Schneichen[H]. Geh hinauf!»

  [H] Seidene Tücher.

Das Licht wurde zurückgezogen, und der Kopf verschwand.

Jack führte Oliver eine finstere, sehr schadhafte Treppe hinauf, mit
der er jedoch sehr genau bekannt zu sein schien, öffnete die Tür eines
Hinterzimmers und zog Oliver nach.

Die Wände des Gemachs waren von Schmutz und Rauch geschwärzt, auf einem
elenden Tische stand ein in den Hals einer Bierflasche gestecktes Licht
und am Kamine die zusammengeschrumpfte Gestalt eines alten Juden mit
einem zurückstoßenden, spitzbübischen, satanischen Gesicht, das durch
dichte, klebrige, rote Haare verdunkelt wurde. Er steckte in einem
fettigen flanellenen Schlafrocke, trug den Hals bloß und schien seine
Aufmerksamkeit zwischen dem Feuer, an welchem er Brotschnitte röstete,
und dem Kleidergestell zu teilen, auf welchem eine große Anzahl
seidener Taschentücher hing. An dem Tische saßen vier oder fünf Knaben,
keiner älter als Jack, rauchten aus langen Tonpfeifen und tranken
Branntwein, ganz als wenn sie Erwachsene gewesen wären. Sie drängten
sich um den Baldowerer, als er dem Juden einige Worte zuflüsterte,
drehten sich darauf nach Oliver um, und sie und der Jude grinsten ihn
an.

«Fagin, das ist er, mein Freund Oliver Twist», sagte Jack Dawkins laut.

Der Jude grinste, machte Oliver eine tiefe Verbeugung, faßte seine
Hand und sagte, er hoffe, die Ehre seiner näheren Bekanntschaft zu
haben. Hierauf umringten ihn die jungen, rauchenden Gentlemen und
drückten ihm eifrig die Hände -- besonders die linke, in welcher er
sein kleines Bündel trug. Der eine von ihnen zeigte großen Eifer,
seine Kappe aufzuhängen, und ein anderer war so dienstfertig, in
seine Tasche zu greifen, um ihn der Mühe zu überheben, wenn er sich
niederlegte, sie auszuleeren; und alle diese Höflichkeiten würden kein
Ende gehabt haben, wenn der Jude die Köpfe und Schultern der gefälligen
jungen Herren nicht mit der Röstgabel, die er in der Hand hielt, zu
bearbeiten angefangen hätte.

«Wir sind alle sehr erfreut, dich kennen zu lernen, Oliver», sagte
der Jude. «Baldowerer, mache einen Platz für Oliver am Feuer frei.
Ah, du betrachtest verwundert die Taschentücher, mein Lieber? Nicht
wahr, es sind ihrer eine ganze Menge? Wir haben sie soeben zum Waschen
herausgehängt. Das ist alles, Oliver; das ist alles. Ha, ha, ha!»

Seine letzten Worte wurden von einem schallenden Gelächter all der
hoffnungsvollen Zöglinge des lustigen alten Herrn begrüßt, worauf sich
alle zu Tisch setzten.

Nachdem Oliver seinen Teil gegessen, mischte ihm der Jude ein Glas
heißen Genever mit Wasser und sagte ihm, er müsse sogleich austrinken,
weil noch jemand des Glases bedürfe. Oliver tat, was ihm geheißen
war, sein Freund Jack hob ihn auf, legte ihn auf ein aus alten Säcken
bereitetes Lager, und er versank sogleich in einen tiefen Schlummer.




9. Kapitel.

    Weitere Mitteilungen über den alten Herrn und seine hoffnungsvollen
    Zöglinge.


Es war schon spät am folgenden Morgen, als Oliver aus einem langen,
festen Schlummer erwachte, doch vorerst nur zu jenem Mittelzustande
zwischen Schlaf und Wachen, in welchem man sich noch nicht vollkommen
ermuntern kann und doch alles hört und sieht, was umher vorgeht.

Der Jude war außer Oliver allein im Zimmer. Er schlürfte seinen
Kaffee, setzte das Geschirr nach einiger Zeit zur Seite, stand eine
Weile am Kamin, wie wenn er nicht wüßte, was er zunächst vornehmen
sollte, blickte darauf nach Oliver hin und rief ihn beim Namen. Oliver
antwortete nicht und schien noch zu schlafen.

Der Jude horchte, ging zur Tür, schob den Riegel vor und nahm darauf,
wie es Oliver schien, aus einer Vertiefung des Fußbodens eine kleine
Schachtel heraus und stellte sie auf den Tisch. Seine Augen glänzten,
als er sie öffnete und in die Schachtel hineinschaute. Er setzte sich
und nahm eine goldene, von Diamanten funkelnde Uhr heraus.

«Aha!» murmelte er mit einem entsetzlichen Lächeln. «Verdammt pfiffige
Bestien! Und courageux bis zum letzten Augenblick. Sagten mit keinem
Sterbenswörtchen dem alten Pfarrer, wo sie wären, verkappten[I] den
alten Fagin nicht. Und was hätt's ihnen geholfen? Der Strick wäre doch
geblieben fest -- hätten gebaumelt keinen Augenblick später. Nein,
nein! Wackre Bursche, wackre Bursche!»

  [I] Verraten.

Er legte die Uhr wieder in die Schachtel, nahm mehrere andere und dann
Ringe, Armbänder und viele Kostbarkeiten heraus, deren Namen oder
Gebrauch Oliver nicht einmal kannte, und beäugelte sie mit gleichem
Vergnügen. Hierauf legte er ein sehr kleines Geschmeide in seine
flache Hand und schien lange bemüht, zu lesen, was darin eingegraben
sein mochte. Endlich ließ er es, wie am Erfolge verzweifelnd, wieder
in die Schachtel hineinfallen, lehnte sich zurück und murmelte: «Was
es doch ist für 'ne hübsche Sache ums Hängen! Tote bereuen nicht --
bringen ans Licht keine dummen Geschichten. Selbst die Aussicht auf den
Galgen macht sie keck und dreist. 's ist sehr schön fürs Geschäft. Fünf
aufgehangen in einer Reihe, und keiner übrig zu teilen mit mir oder zu
lehmern[J].»

  [J] Verraten, beichten.

Er blickte auf, seine schwarzen, stechenden Augen begegneten Olivers
Blicken, die in stummer Neugier auf ihn geheftet waren, und er gewahrte
sogleich, daß er beobachtet worden war. Er drückte die Schachtel zu,
griff nach einem auf dem Tische liegenden Messer und sprang wütend und
am ganzen Leibe zitternd auf.

«Was ist das?» rief er. «Warum passest du mir auf? Warum bist du wach?
Was hast du gesehen? Sprich, Bube -- sprich, sprich, so lieb dir dein
Leben ist!»

«Ich konnte nicht mehr schlafen», erwiderte Oliver bestürzt. «Es tut
mir sehr leid, wenn ich Sie gestört habe, Sir!»

«Hast du nicht schon seit einer Stunde gewacht?» fragte der Jude,
Oliver finster anblickend.

«Nein, Sir -- nein, wahrlich nicht», sagte Oliver.

«Ist's auch wahr?» rief der Jude mit noch drohenderen Gebärden.

«Auf mein Wort, Sir!» versicherte Oliver.

«Schon gut, schon gut!» fuhr der Jude, auf einmal sein gewöhnliches
Wesen wieder annehmend, fort. «Ich weiß es wohl -- wollte dich nur
erschrecken -- auf die Probe stellen. Du bist ein wackerer Junge,
Oliver.» Er rieb sich kichernd die Hände, blickte jedoch unruhig nach
der Schachtel hin. «Hast du gesehen die hübschen Sachen?» fragte er
nach einigem Stillschweigen.

«Ja, Sir.»

«Ah!» rief erblassend der Jude aus. «Sie -- sind mein Eigentum, Oliver;
mein kleines Eigentum -- alles, was ich besitze für meine alten Tage.
Man schilt mich einen Geizhals -- aber ich muß doch leben.»

Oliver dachte, der alte Herr müsse wirklich ein Geizhals sein, denn er
würde sonst nicht, obgleich im Besitz solcher Schätze, so erbärmlich
wohnen. Indes meinte er, seine Liebe zu Jack und den anderen Knaben
möchte ihm wohl viel Geld kosten. Er fragte schüchtern, ob er aufstehen
dürfe. Der Jude hieß ihn Wasser zum Waschen aus dem dastehenden
Steinkruge holen, und als Oliver es geschöpft hatte und sich umdrehte,
war die Schachtel verschwunden.

Er hatte sich kaum gewaschen, als der Baldowerer nebst einem der Knaben
eintrat, die Oliver am vorigen Abend hatte rauchen sehen. Jack stellte
ihm seinen Begleiter, Charley Bates, förmlich vor, und alle vier
setzten sich zum Frühstück, das Jack in seinem Hute mitgebracht hatte.

«Ich hoffe, daß ihr heute morgen gearbeitet habt!» sagte der Jude zu
Jack, nach Oliver blinzelnd.

«Tüchtig!» lautete die Antwort.

«Wie Drescher!» setzte Charley Bates hinzu.

«Ah, ihr seid gute Jungen! Was hast du mitgebracht, Baldowerer?»

«Ein paar Brieftaschen!» erwiderte Jack und reichte ihm eine rote und
eine grüne hin.

Der Jude öffnete beide und durchsuchte sie mit bebender Begier. «Nicht
so schwer, als sie sein könnten», bemerkte er; «aber doch artige
Arbeit, recht artige Arbeit -- nicht wahr, Oliver?»

«Ja, wahrlich, Sir!» antwortete Oliver, worüber Charley Bates, zur
großen Verwunderung Olivers, laut zu lachen anfing.

«Was hast du denn mitgebracht, Charley?» fragte der Jude.

«Schneichen!» erwiderte Master Bates und wies vier Taschentücher vor.

Der Jude nahm sie in genauen Augenschein.

«Sie sind sehr gut», sagte er; «du hast sie aber nicht gezeichnet gut;
die Buchstaben müssen wieder ausgelöst werden, und das soll Oliver
lernen. Willst du, Oliver?»

«Wenn Sie es befehlen, gern, Sir!» war Olivers Antwort.

«Möchtest du mir wohl ebenso leicht Taschentücher anschaffen können wie
Charley?»

«Warum nicht -- wenn Sie es mich lehren wollen, Sir?»

Charley brach abermals in ein schallendes Gelächter aus und wäre dabei
fast erstickt, da er eben einen Bissen zum Munde geführt hatte. «Er ist
gar zu allerliebst grün!» rief er endlich, gleichsam zur Entschuldigung
seines unhöflichen Benehmens, aus.

Der Baldowerer bemerkte, Oliver würde seinerzeit schon alles lernen.
Der Jude sah Oliver die Farbe wechseln und lenkte das Gespräch auf
einen anderen Gegenstand. Er fragte, ob viele Zuschauer bei der
Hinrichtung gewesen wären, und Olivers Erstaunen wuchs immer mehr,
denn aus den Antworten Jacks und Charleys ging hervor, daß sie
beide zugegen gewesen waren, und es war ihm unerklärlich, wie sie
dessenungeachtet so fleißig hatten arbeiten können.

Als das Frühstück beendet war, spielten der muntere alte Herr und die
beiden Knaben ein äußerst sonderbares und ungewöhnliches Spiel. Der
alte Herr steckte eine Dose, eine Brieftasche und eine Uhr in seine
Taschen, eine Brustnadel in sein Hemd, hing eine Uhrkette um den Hals,
knöpfte den Rock dicht zu, ging auf und ab, blieb bisweilen stehen,
als wenn er in einen Laden hineinsähe, blickte beständig umher, als
wenn er Furcht vor Dieben hegte, befühlte seine Taschen, wie um sich
zu überzeugen, ob er auch nichts verloren hätte, und machte das alles
so spaßhaft und natürlich, daß Oliver lachte, bis ihm die Tränen über
die Wangen hinabliefen. Die beiden Knaben verfolgten unterdes den
Alten und entschwanden, wenn er sich umdrehte, seinen Blicken mit der
bewunderungswürdigsten Behendigkeit. Endlich trat ihm der Baldowerer
wie zufällig auf die Zehen, während Charley Bates von hinten gegen ihn
anrannte, und sie entwendeten ihm dabei Taschentuch, Uhr, Brustnadel
usw. so geschickt, daß Oliver kaum ihren Bewegungen zu folgen
vermochte. Fühlte der alte Herr eine Hand in einer seiner Taschen, so
war der Dieb gefangen, und das Spiel fing von vorn wieder an.

Es war mehreremal durchgespielt, als zwei junge Damen erschienen, um
die jungen Herren zu besuchen. Die eine hieß Betsy, die andere Nancy.
Ihr Haar war nicht in der genauesten Ordnung, ihre Schuhe und Strümpfe
schienen nicht im besten Zustande zu sein. Sie waren vielleicht nicht
eigentlich schön, hatten aber viel Farbe und ein kräftiges, munteres
Aussehen. Ihre Manieren waren sehr frei und angenehm, und so meinte
Oliver, daß sie sehr artige Mädchen wären, was sie auch ohne Zweifel
waren.

Sie blieben lange. Es wurden geistige Getränke gebracht, da die jungen
Damen über innerliche Kälte klagten, und die munterste Unterhaltung
entspann sich. Endlich erinnerte sich Charley Bates, daß es Zeit sei,
auszugehen. Der gute alte Herr gab ihm und dem Baldowerer verschiedene
Anweisungen und Geld zum Ausgeben, worauf sie sich nebst Betsy und
Nancy entfernten.

«Ist's nicht ein angenehmes Leben, das meine Knaben führen?» sagte
Fagin.

«Sind sie denn auf Arbeit ausgegangen?» fragte Oliver.

«Allerdings», erwiderte der Jude; «und sie arbeiten den ganzen Tag
unverdrossen, wenn sie nicht werden gestört. Nimm sie dir zum Muster,
mein Kind; tu alles, was sie dir heißen; und folg' jederzeit ihrem Rat,
besonders dem des Baldowerers. Er wird werden ein großer Mann und auch
aus dir machen 'nen großen Mann, wenn du dir ihn zum Vorbilde nimmst.
Hängt mein Taschentuch aus der Tasche, mein Lieber?»

«Ja, Sir!» sagte Oliver.

«So sieh einmal zu, ob du es herausziehen kannst, ohne daß ich's fühle,
wie du's vorhin gesehen hast von den beiden.»

Oliver erinnerte sich genau, wie er es Jack hatte tun sehen, und tat es
ihm nach.

«Ist's heraus?»

«Hier ist es, Sir.»

«Du bist ein kluger Knabe», sagte der alte Herr, ihm die Wange
klopfend; «ich habe niemals gesehen ein anstelligeres Kind. Da hast du
'nen Schilling. Fährst du so fort, so wirst du werden der größte Mann
deiner Zeit. Doch will ich dir jetzt zeigen, wie man herauslöst die
Buchstaben.»

Oliver konnte gar nicht begreifen, wie er ein großer Mann dadurch
werden könne, daß er dem alten Herrn das Tuch aus der Tasche zöge,
meinte jedoch, daß es der so viel ältere besser wissen müsse als er,
und war bald eifrig mit seinen neuen Studien beschäftigt.




10. Kapitel.

    Oliver gewinnt Erfahrung um einen hohen Preis.


Oliver blieb acht bis zehn Tage im Zimmer des Juden, wurde fortwährend
beschäftigt, Zeichen aus den Taschentüchern, von denen eine große Menge
nach Hause gebracht wurde, herauszutrennen, und nahm bisweilen an dem
beschriebenen Spiele teil, das täglich gespielt wurde. Er fing immer
mehr an, sich nach frischer Luft zu sehnen, und bat den alten Herrn
mehrmals auf das dringendste, ihn mit seinen beiden Kameraden zum
Arbeiten ausgehen zu lassen.

Endlich wurde ihm eines Morgens die Erlaubnis erteilt, unter Jacks und
Charleys Aufsicht auszugehen. Es waren keine Taschentücher mehr da,
an denen Oliver hätte arbeiten können, und vielleicht war dies der
Grund, weshalb der alte Herr seine Zustimmung gab. Die Knaben gingen
und gerieten sogleich in ein sehr langsames Schlendern, was Oliver
höchst mißbilligte, eingedenk der vielfachen Warnungen des alten Herrn
vor dem verderblichen Müßiggange. Der Baldowerer verübte mannigfachen
Mutwillen an Knaben, und Charley erlaubte sich sogar, die Heiligkeit
des Eigentums zu verletzen, wenn er an einem Apfel- oder Zwiebelkorbe
vorüberkam. Oliver war daher schon im Begriff, unwillig heimzukehren,
als seine Begleiter auf einmal anfingen, sich äußerst geheimnisvoll zu
benehmen, wodurch er von seinem Vorhaben abgelenkt wurde.

Sie umschlichen einen alten Herrn, auf den sie ihn aufmerksam gemacht
hatten, ohne seine Fragen anders als durch einige ihm unverständliche
Worte und Winke zu beantworten. Er hielt sich einige Schritte hinter
ihnen und stand endlich, unschlüssig, ob er weitergehen oder sich
zurückziehen solle, verwundert zuschauend da.

Der alte Herr sah sehr respektabel aus, trug Puder in den Haaren und
eine goldene Brille. Er hatte sich vor einen Bücherladen hingestellt,
ein Buch zur Hand genommen, las darin, sein spanisches Rohr unter dem
linken Arme, und hörte und sah offenbar nicht, was um ihn her vorging.

Wer beschreibt Olivers Bestürzung, als der Baldowerer dem alten Herrn
das Tuch aus der Tasche zog, es Charley Bates reichte, und als darauf
beide spornstreichs davonliefen! Im Augenblick war ihm das Geheimnis
der Taschentücher, Uhren und Kleinodien klar. Das Blut stockte ihm
in den Adern, ihm schwindelte vor Furcht und Schrecken, und ohne zu
wissen, was er tat, lief er seinen Kameraden nach, so schnell seine
Füße ihn tragen mochten. In demselben Augenblick griff der alte Herr
nach seinem Tuche in die Tasche, vermißte es, drehte sich rasch um, sah
Oliver laufen und erhob den Ruf: «Halt den Dieb!» -- den magischen Ruf,
auf welchen sofort alles lebendig wird, der Krämer aus seinem Laden
auf die Straße stürzt, der Gemüsehändler seinen Korb, der Milchmann
seinen Eimer, der Pflasterer seine Ramme, der Schulknabe seine Bücher
im Stiche läßt und alles nachläuft.

Jack und Charley hatten Aufsehen zu vermeiden gewünscht und waren
daher nur bis um die nächste Ecke gelaufen, worauf sie sich unter
einem Torwege neugierigen Blicken zu entziehen suchten. Sobald sie das
Geschrei «Halt den Dieb!» vernahmen, stimmten sie aus allen Kräften ein
und schlossen sich wie gute Bürger den Verfolgern an. Diese Anwendung
des großen Naturgesetzes der Selbsterhaltung war Oliver vollkommen neu.
Er wurde noch mehr verwirrt und bestürzt und verdoppelte seine Eile,
sah sich indes nach einiger Zeit eingeholt und wurde obendrein zu Boden
geschlagen.

In wenigen Augenblicken war ein zahlreicher Haufen um ihn versammelt.
«Drückt ihn doch nicht tot!» -- «Verdient er's besser?» -- «Wo ist der
bestohlene Herr?» -- «Da kommt er schon; macht Raum für den Herrn!» --
«Ist dies der Bursch, Sir?» -- «Ja!»

Oliver lag da, mit Schmutz bedeckt, blutend aus Nase und Mund, und sah
betäubt und geängstet umher.

«Ich fürchte, daß es der Knabe ist», sagte der Herr sehr milde.

«Das fürchten Sie? Der ist auch wohl der Rechte.»

«Der arme Kleine hat sich beschädigt!» fuhr der Herr fort.

«Das hab' ich getan», fiel ein vierschrötiger Mensch, hervortretend,
ein; «traf ihn gerade mit der Faust auf die Schnauze -- ich hab' ihn
aufgehalten für Sie, Sir.»

Er zog grinsend den Hut, eine Belohnung seiner Dienstfertigkeit
erwartend; allein der alte, dicke Herr blickte ihn unwillig an und
hätte sich offenbar gern entfernt, wenn sich nicht ein Polizist, der in
solchen Fällen gewöhnlich zuletzt kommt, in diesem Augenblick durch die
Menge gedrängt und Oliver beim Kragen gepackt hätte.

«Steh auf!» sagte der Mann barsch.

«Ich bin es wirklich nicht gewesen, Sir, wirklich und wahrhaftig
nicht. Es waren zwei andere Knaben», sagte Oliver, die Hände bittend
zusammenlegend. «Sie müssen hier irgendwo in der Nähe sein.»

«O nein, sie sind nicht hier», entgegnete der Beamte. Er meinte dies
ironisch, aber es war die volle Wahrheit, denn der Baldowerer und
Charley Bates hatten sich längst aus dem Staube gemacht. «Steh auf!»

«Tun Sie ihm nichts zuleide», sagte der menschenfreundliche Herr.

«O nein, ich werde ihm nichts zuleide tun», erwiderte der Polizist,
indem er zum Beweise dafür Oliver die Jacke halb vom Rücken riß. «Komm
nur; ich kenne dich schon. Willst du mal auf deinen Füßen stehen,
verdammter kleiner Strolch!»

Oliver machte einen Versuch, sich zu erheben, konnte sich aber kaum
aufrecht erhalten und wurde am Kragen seiner Jacke im Laufschritt
durch die Straßen geschleppt. Der alte Herr ging mit, und ein immer
anwachsender Volkshaufen folgte johlend und lärmend den drei nach der
nächsten Polizeiwache.




11. Kapitel.

    Wie Mr. Fang die Gerechtigkeit handhabte.


Der Diebstahl war im Bezirke dieses Polizeiamtes begangen worden.
Als der Zug auf der Wache anlangte, wurde Oliver vorläufig in ein
kellerartiges Gemach eingeschlossen, das über alle Beschreibung
schmutzig war, denn sechs Betrunkene hatten es fast drei Tage
inne gehabt. Doch das will nichts sagen. Sperrt man doch Tag für
Tag und Nacht für Nacht Männer und Weiber um der geringfügigsten,
leichtfertigsten Anschuldigungen willen in Spelunken ein, gegen welche
die Zellen der schwersten und bereits verurteilten Verbrecher im
Newgategefängnisse für Prunkgemächer gelten könnten!

Der alte Herr sah Oliver mitleidig und wehmütig nach. --

«Es liegt ein Ausdruck in den Zügen des Knaben, der mich ganz wunderbar
ergreift», sprach er bei sich selbst. «Sollte er nicht unschuldig sein?
Er sah aus, als wenn er -- hm! -- ist mir's doch in der Tat, als wenn
ich dieses Gesicht oder ein ganz ähnliches schon gesehen hätte.»

Er sann und sann, rief sich die Züge seiner Freunde, Feinde und
Bekannten, alter und neuer, längst vergessener, längst im Grabe
ruhender ins Gedächtnis zurück, vermochte sich aber dennoch auf keines
zu entsinnen, mit welchem Oliver Ähnlichkeit gehabt hätte. «Nein, es
muß Einbildung sein», sagte er endlich seufzend und kopfschüttelnd.

Er wurde durch eine Berührung an der Schulter aus seinem Sinnen
aufgeschreckt und bemerkte, als er sich umwandte, den Schließer, der
ihn aufforderte, ihm ins Amtszimmer zu folgen. Als er eintrat, saß Mr.
Fang, der Polizeirichter, bereits hinter einer Barriere am oberen Ende,
und neben der Tür befand sich eine Art von hölzernem Verschlag, in dem
der arme Oliver, an allen Gliedern zitternd, hockte. Mr. Fangs Antlitz
hatte den Ausdruck der Härte und war sehr rot. Wenn er nicht mehr zu
trinken pflegte, als ihm gut war, so hätte er gegen sein Gesicht eine
Injurienklage anstellen können, und sicher würden ihm beträchtliche
Entschädigungsgelder zuerkannt worden sein.

Der alte Herr verbeugte sich ehrerbietig.

«Hier ist mein Name und meine Adresse, Sir!» sagte er und reichte Mr.
Fang seine Karte.

Mr. Fang, der eben seine Zeitung las, war unwillig über die Störung und
blickte ärgerlich auf.

«Wer sind Sie?»

Der alte Herr wies ein wenig erstaunt auf seine Karte.

Mr. Fang stieß sein Zeitungsblatt nebst der Karte verächtlich zur Seite.

«Gerichtsdiener! Wer ist dieser Mensch?»

«Sir, ich heiße Brownlow», fiel der alte Herr mit dem Anstande eines
Gentleman in starkem Kontrast zu Mr. Fang ein. «Erlauben Sie, daß ich
um den Namen des Richters bitte, der einen anständigen Mann ohne alle
Veranlassung im Gerichtslokale beleidigt.»

«Gerichtsdiener!» herrschte Fang; «wessen ist dieser Mensch angeklagt?»

«Er ist nicht angeklagt, Ihr Edeln, sondern erscheint als Ankläger des
Knaben.»

Seine Edeln wußten das sehr wohl, konnten jedoch auf die Weise ganz
sicher unangenehme Dinge sagen.

«Erscheint als Ankläger des Knaben -- so!» sagte Fang, Brownlow
verächtlich von Kopf bis zu den Füßen betrachtend. «Nehmen Sie ihm den
Eid ab.»

«Bevor das geschieht, muß ich mir ein paar Worte erlauben», fiel
Brownlow ein. «Ich würde nämlich, ohne daß es mir wirklich widerfahren
wäre, niemals geglaubt haben --»

«Halten Sie den Mund, Sir!» unterbrach ihn Fang in befehlshaberischem
Tone.

«Ich will und werde reden!» sagte Brownlow ebenso bestimmt.

«Sie halten augenblicklich den Mund, Sir, oder ich lasse Sie
hinausbringen. Sie sind ein unverschämter Mensch! Wie können Sie es
wagen, sich den Anordnungen eines Richters widersetzen zu wollen?»

Dem alten Herrn stieg das Blut ins Gesicht.

«Vereidigen Sie dieses Individuum!» rief Fang dem Schreiber zu. «Ich
will durchaus nichts mehr hören.»

Brownlow war im höchsten Grade entrüstet, glaubte aber, dem Knaben
möglicherweise schaden zu können, wenn er seine Gefühle nicht
unterdrückte, und legte daher den Eid ab.

«Wohin geht Ihre Anklage?» fragte ihn Fang darauf. «Was haben Sie zu
sagen, Sir?»

«Ich stand vor einem Bücherladen», begann Brownlow, allein Fang
unterbrach ihn.

«Schweigen Sie, Sir. Wo ist der Polizist? Vereidigen Sie den
Polizisten. Polizist -- reden Sie!»

Der Polizist berichtete mit gebührender Unterwürfigkeit, wie er den
Knaben gefunden, und wie er ihm die Taschen durchsucht und nichts
gefunden habe; -- mehr wisse er nicht.

«Sind Zeugen vorhanden?» fragte Fang.

«Nein, Ihr Edeln.»

Fang saß ein paar Minuten schweigend da, wendete sich darauf zu
Brownlow und sagte in großer Hitze: «Denken Sie Ihre Anklage gegen den
Knaben anzubringen oder nicht? Sie haben geschworen. Verweigern Sie Ihr
Zeugnis, so werd' ich Sie wegen Nichtachtung der Richterbank in Strafe
nehmen; das werd' ich, beim --»

Es ist und bleibt unbekannt, bei wem; denn der Schreiber hustete im
rechten Augenblick und ließ ein Buch zur Erde fallen -- natürlich nur
zufällig.

Brownlow konnte endlich vorbringen, was er zu sagen hatte, und fügte
hinzu, daß er die Hoffnung hege, der Richter werde die Gesetze so mild
wie möglich anwenden, wenn er es als erwiesen annehmen sollte, daß der
Knabe, wenn er nicht selbst ein Dieb sei, doch mit Dieben in Verbindung
stehe.

«Er ist bereits hart beschädigt,» schloß er, «und ich fürchte, daß ihm
sehr unwohl ist.»

«Unwohl -- so, so!» sagte Fang mit einem höhnischen Lächeln. «Du
spielst mir hier keine Komödie, du kleiner Landstreicher, das sag' ich
dir; kommst mir damit nicht durch. Wie heißest du?»

Oliver wollte antworten, aber die Zunge versagte den Dienst. Er war
totenblaß, und alles schien sich mit ihm zu drehen.

«Wie heißest du, du verhärteter Schlingel?» donnerte ihn Fang
wiederholt an. «Gerichtsdiener, wie heißt der Bube?»

Der Gerichtsdiener beugte sich über Oliver und wiederholte die Frage,
gewahrte aber, daß der Knabe wirklich nicht imstande war zu antworten,
und sagte daher, weil er wußte, daß der Richter sonst nur noch wütender
werden und eine noch härtere Strafe diktieren würde: «Er sagt, sein
Name wäre Tom White, Ihr Edeln.»

«Wo wohnt er?» fragte Fang weiter.

«Wo er eben kann!» erwiderte der gutherzige Gerichtsdiener abermals für
Oliver.

«Hat er Eltern?»

«Er sagt, sie wären in seiner Kindheit gestorben, Ihr Edeln!»
entgegnete der Gerichtsdiener. Es war die gewöhnliche Antwort in Fällen
dieser Art.

Oliver hob bei der letzten Frage den Kopf empor, sah mit flehenden
Blicken umher und bat mit schwacher Stimme um ein Glas Wasser.

«Albernheiten!» sagte Fang. «Hab' mich ja nicht zum Narren, Bursch!»

«Ich glaube wirklich, daß ihm unwohl ist, Ihr Edeln!» wendete der
Gerichtsdiener ein.

«Ich weiß es besser», fuhr Fang auf.

«Gerichtsdiener, halten Sie ihn!» rief der alte Herr, «oder er sinkt zu
Boden.»

«Zurück da, Gerichtsdiener!» tobte Fang; «mag er, wenn's ihm beliebt.»

Oliver bediente sich der freundlichen Erlaubnis und fiel ohnmächtig von
seiner Bank herunter.

Der Richter befahl, ihn liegen zu lassen, bis er wieder zu sich käme;
der Schreiber fragte leise, wie Mr. Fang zu verfahren gedächte.

«Summarisch», erwiderte Mr. Fang. «Er wird drei Monate eingesperrt --
natürlich bei harter Arbeit.»

Zwei Schließer schickten sich an, den ohnmächtigen Knaben in seine
Zelle zu tragen, als plötzlich ein ältlicher, ärmlich, aber anständig
gekleideter Mann atemlos hereintrat.

«Halt -- halt!» rief er; «um des Himmels willen noch einen Augenblick
Geduld.»

Obgleich die Polizeibeamten die willkürlichste Gewalt über die
Freiheit, den guten Ruf und Namen, ja fast das Leben der königlichen
Untertanen, besonders der ärmeren Klassen, zu üben pflegen, und
obgleich in den Polizeigerichten genug Dinge vorgehen, um den Engeln
blutige Tränen auszupressen, so erfährt das Publikum doch nichts davon,
ausgenommen durch das Medium der Tagespresse. Mr. Fang war daher nicht
wenig entrüstet, einen ungebetenen Gast eintreten und so ordnungswidrig
auftreten zu sehen.

«Was ist das? Wer ist das? Werft den Menschen hinaus!» rief er.

«Ich will und muß reden, Sir; ich lasse mich nicht hinauswerfen; hab's
alles angesehen. Ich bin der Besitzer des Buchladens. Ich verlange,
vereidigt zu werden. Mr. Fang, Sie müssen mich anhören -- Sie können es
nicht wagen, mein Zeugnis zurückzuweisen, Sir.»

Er war im Recht und sah zu entschlossen aus, als daß der Richter
es hätte wagen dürfen, ihn abzuweisen. Fang ließ ihm daher den Eid
abnehmen und fragte darauf, was er zu sagen habe.

«Ich sah drei Knaben -- zwei andere und diesen hier -- um den Herrn
da herumschleichen, der vor meinem Laden stand und las. Der Diebstahl
wurde von einem anderen Knaben begangen, und dieser war ganz erstaunt
darüber -- sah aus, als wenn ihn der Schlag gerührt hätte.»

«Warum kamen Sie nicht schon früher her?»

«Ich hatte niemand, nach meinem Laden zu sehen, und bin hergelaufen,
sobald ich jemand auftreiben konnte.»

«Also der Ankläger las?»

«Ja, Sir -- in dem Buche, das er in diesem Augenblicke in der Hand hat.»

«Ah -- ist es bezahlt?»

«Nein!» erwiderte der Buchhändler lächelnd.

«Mein Himmel, das hab' ich ganz vergessen!» rief der zerstreute alte
Herr ganz unbefangen aus.

«Vortrefflich! -- Und Sie werfen sich zum Ankläger eines unglücklichen,
armen Knaben auf!» bemerkte Fang mit komisch aussehender Anstrengung,
eine menschenfreundliche Miene anzunehmen. «Es scheint mir, Sir, daß
Sie unter sehr verdächtigen und unehrenhaften Umständen zu dem Buche
gelangt sind, und Sie können sich sehr glücklich schätzen, wenn der
Eigentümer nicht als Ankläger gegen Sie auftreten will. Nehmen Sie
sich dies zur Lehre, mein Freund, oder Sie verfallen noch einmal dem
Gesetze. Der Knabe ist freizulassen. Räumen Sie das Gerichtszimmer!»

Der alte Herr wurde unter Ausbrüchen der Entrüstung, die er nicht
länger mehr zurückzuhalten vermochte, hinausgeführt. Er stand im
Hofraume, und sein Zorn verschwand. Oliver lag auf dem Steinpflaster;
man hatte ihm die Schläfe mit Wasser gewaschen; er war weiß wie eine
Leiche und zitterte krampfhaft am ganzen Leibe. «Armes Kind, armes
Kind!» sagte Mr. Brownlow, sich über ihn hinunterbeugend. «Leute, ich
bitte, schaff' mir doch jemand sogleich einen Mietwagen.»

Gleich darauf fuhr ein leerer Wagen vorüber, Oliver wurde sorgfältig
hineingehoben und auf einen Sitz gelegt, während der alte Herr auf dem
anderen Platz nahm.

«Darf ich Sie begleiten?» fragte der Buchhändler.

«Ja, ja, mein werter Herr!» erwiderte Brownlow. «Ich habe Sie
vergessen; verzeihen Sie. Und da hab' ich auch das unglückliche Buch
noch. Steigen Sie geschwind ein, es ist keine Zeit zu verlieren.»

Der Buchhändler setzte sich zu Brownlow, und sie fuhren ab.




12. Kapitel.

    In welchem für Oliver bessere Fürsorge getragen wird, als er sie
    noch in seinem ganzen Leben erfahren. Die Geschichte kehrt zu dem
    lustigen alten Herrn und seinen hoffnungsvollen Zöglingen zurück.


Der Wagen hielt nach ziemlich langer Fahrt vor einem hübschen Hause in
einer stillen Straße, nicht weit von Pentonville. Mr. Brownlow ließ
Oliver sogleich zu Bett bringen und sorgte mit einem Eifer für Pflege
jeder Art, der keine Grenzen kannte. Sein Schützling verfiel in ein
heftiges Fieber und erwachte erst nach acht Tagen aus einem langen
und unruhigen Traume, wie es ihm schien. «Wo bin ich?» rief er mit
schwacher Stimme. «Wer hat mich hierher gebracht?»

Der Vorhang seines Bettes wurde rasch zurückgeschoben, und eine
mütterlich aussehende, sauber gekleidete alte Frau beugte sich über
ihn und sagte: «Ruhig, mein Söhnchen, du mußt ganz still liegen oder
wirst sonst wieder krank werden. Denn du hast an der Schwelle des Todes
gestanden; also verhalte dich ja recht ruhig.»

Sie sah so freundlich und liebevoll dabei aus und strich ihm so
sorglich das Haar von der Stirn zurück, daß er sich nicht enthalten
konnte, seine abgezehrte Hand auf die ihrige zu legen und einige, wenn
auch unverständliche Worte gerührten Dankes zu murmeln.

«Was es für ein lieber Kleiner ist!» sagte sie mit Tränen in den Augen.
«Wie würde sich seine Mutter freuen, wenn sie so wie ich bei ihm
gesessen hätte und ihn jetzt sähe!»

«Vielleicht sieht sie mich,» flüsterte Oliver und faltete seine Hände.
«Vielleicht war sie bei mir, Ma'am. Es ist mir fast, als wäre sie hier
gewesen.»

«Das macht das Fieber, mein Kind», bemerkte Frau Bedwin.

«Kann wohl sein», erwiderte Oliver nachdenklich; «denn der Himmel ist
sehr fern, und die Seligen haben es dort zu gut, als daß sie an das
Krankenbett eines armen Knaben herunterkommen sollten. Wenn sie es aber
gewußt hat, daß ich krank war, so hat sie gewiß Mitleid mit mir gehabt,
denn sie war selbst sehr krank, ehe sie starb. Aber -- sie mag wohl
nichts von mir wissen, denn wenn sie mich hätte niederschlagen sehen,
so würde sie sehr betrübt geworden sein, und ihr Gesicht war immer so
froh und vergnügt, wenn ich von ihr geträumt habe.»

Frau Bedwin wischte sich die Augen, brachte ihm zu trinken und ermahnte
ihn abermals, ganz still zu liegen, weil er sonst wieder krank werden
würde. Er schwieg daher und hielt sich vollkommen ruhig, teils weil
er der guten Frau nicht ungehorsam sein wollte, und andernteils, weil
er durch das, was er gesagt hatte, bereits vollkommen erschöpft war.
Er schlief ein, und als er erwachte, stand ein Herr an seinem Bette,
der seinen Puls fühlte. «Nicht wahr, mein Kind, du fühlst dich weit
besser?» fragte ihn der Herr.

«Ja, ich danke, Sir!» antwortete Oliver.

«Das wußte ich wohl. Und du bist hungrig -- nicht wahr?»

«Nein, Sir.»

«Hm! Ja, ganz recht. Du kannst auch in der Tat keinen Hunger empfinden.
Er ist nicht hungrig, Frau Bedwin», sagte der Herr mit sehr weiser
Miene.

Frau Bedwin neigte ehrfurchtsvoll den Kopf, wodurch sie andeuten zu
wollen schien, daß sie den Doktor für einen äußerst gescheiten Mann
hielte. Der Doktor schien vollkommen derselben Meinung zu sein.

«Du bist müde, nicht wahr, mein Sohn?» sagte er.

«Nein, Sir.»

«Nicht?» wiederholte der Doktor; «das freut mich, und ich dachte es
wohl. Aber durstig bist du?»

«Ach ja, Sir», erwiderte Oliver.

«Ganz wie ich es erwartet habe. Frau Bedwin, es ist sehr natürlich, daß
er Durst fühlt. Sie können ihm ein wenig Tee mit Weißbrot ohne Butter
geben. Halten Sie ihn nicht zu warm, Ma'am, und haben Sie acht, daß er
nicht zu kalt wird.»

Frau Bedwin knixte, und der Doktor ging. Oliver schlief bald wieder
ein, und als er erwachte, war es fast zwölf Uhr. Frau Bedwin sagte ihm
gute Nacht und überwies ihn der Pflege einer eingetretenen alten Frau,
die in ihrem Bündel ein kleines Gebetbuch und eine große Nachtmütze
mitgebracht hatte, sich an den Kamin setzte und sehr bald einschlief.

Oliver lag noch einige Zeit wach. Es herrschte eine feierliche Stille,
und als er daran dachte, daß der Tod viele Tage und Nächte über seinem
Bette geschwebt hätte und das Gemach auch wohl noch mit Schmerz und
Wehe erfüllen könnte, begann er inbrünstig zu beten. Er versank darauf
wieder in jenen festen Schlummer, den nur heitere Ruhe nach erduldeten
Leiden gibt und aus welchem man nicht ohne Bedauern erwacht. Wenn es
der Tod wäre -- wer möchte aus ihm wieder aufwachen wollen zu den Mühen
und Ängsten des Lebens, zu den Nöten der Gegenwart, den Sorgen um die
Zukunft, und zumal den trüben Erinnerungen an die Vergangenheit!

Es war heller Tag, als Oliver die Augen aufschlug, er fühlte sich
heiter und froh, die Krise war überstanden, und er gehörte der Welt
wieder an. -- Nach drei Tagen konnte er, durch Kissen gestützt, in
einem Lehnstuhle sitzen. Frau Bedwin ließ ihn in ihr kleines Zimmer
hinunterbringen, setzte sich zu ihm an das Feuer und fing vor Freude
von Herzen zu schluchzen an.

«Sie sind sehr gütig gegen mich, Ma'am», sagte Oliver.

Sie wollte nichts davon hören und bereitete ihm sorglich ein für seinen
Zustand passendes Frühstück. Oliver heftete unterdes seine Blicke auf
ein ihm gerade gegenüber an der Wand hängendes Porträt. Sie wurde
aufmerksam darauf.

«Magst du gern Bilder leiden, mein Kleiner?»

«Ich habe noch wenige gesehen; aber wie schön und liebevoll das Gesicht
der Dame ist!»

«Ah, die Maler machen die Damen immer hübscher, als sie sind,
denn sie würden sonst keine Kundschaft haben. Der Mann, der die
Konterfeimaschine erfand, hätte vorauswissen können, daß es nichts
damit wäre, denn es ist viel zu viel Ehrlichkeit dabei.»

Sie lachte, Oliver aber blieb ernst und fragte: «Wen stellt denn das
Bild vor, Ma'am?»

«Ich weiß es nicht, mein Kind; aber sicher niemand, den wir beide
kennen. Es scheint dir ja erstaunlich zu gefallen.»

«Ach, es ist gar zu schön!» rief Oliver aus.

«Du fängst doch nicht an, dich zu fürchten?» sagte Frau Bedwin, denn
sie gewahrte mit großer Verwunderung, daß Oliver das Porträt mit einer
Art von Beben betrachtete.

«O, nein, nein,» erwiderte er rasch; «aber die Augen blicken so
traurig, und es ist, als wären sie gerade, wo ich sitze, auf mich
geheftet. Es macht mir das Herz schlagen», setzte er mit leiser Stimme
hinzu, «als wenn es lebte und zu mir reden wollte und könnte doch
nicht.»

«Gott sei uns gnädig!» rief Frau Bedwin bestürzt aus; «sprich nicht
so, Kind. Du mußt noch sehr schwach und fieberisch sein. So, so -- nun
kannst du es nicht mehr sehen.»

Sie drehte bei diesen Worten seinen Stuhl herum; Oliver aber sah im
Geiste das Bild so deutlich, als ob es ihm noch immer vor Augen hinge.
Er wollte indes die gute alte Frau nicht ängstigen und lächelte ihr
freundlich zu, als sie ihm seine Brühe mit Weißbrot brachte. Er hatte
kaum einen Löffel voll genossen, als Mr. Brownlow eintrat.

Oliver sah noch sehr blaß und abgezehrt aus; er machte einen
vergeblichen Versuch, aufzustehen, um seinem Wohltäter zu danken, dem
die Tränen in die Augen traten.

«Armes Kind, armes Kind», sagte er. «Wie befindest du dich heute, mein
Lieber?»

«Vortrefflich, Sir», erwiderte Oliver; «und ich bin Ihnen sehr dankbar
für alle Ihre Güte.»

«Gutes Kind,» sagte sein Wohltäter, erkundigte sich darauf, was ihm
Frau Bedwin zur Stärkung gegeben, und bemerkte: «Brühe -- pfui! Ein
paar Gläser Portwein würden ihm besser geschmeckt haben -- nicht wahr,
Tom?»

«Ich heiße Oliver, Sir!» entgegnete der kleine Patient sehr verwundert.

«Oliver! -- Wie? -- Oliver White?»

«Nein, Sir, Twist -- Oliver Twist!»

«Kurioser Name; -- warum sagtest du denn dem Richter, daß du White
hießest?»

«Das hab' ich ihm ganz und gar nicht gesagt», erwiderte Oliver äußerst
verwundert.

Dies sah einer Lüge so ähnlich, daß ihn der alte Herr etwas strenge
ansah. Allein es war unmöglich, seine Aussage zu bezweifeln, denn aus
allen seinen Zügen leuchtete die klarste Wahrheit hervor. Brownlow
meinte, daß ein Mißverständnis obwalten müsse, sein Verdacht schwand
gänzlich, und doch vermochte er die Blicke von Oliver nicht abzuwenden,
denn abermals drängte sich ihm die Ähnlichkeit des Knaben mit bekannten
Zügen auf. Oliver hob flehend die Augen zu ihm empor.

«Sie sind mir doch nicht böse, Sir?»

«Nein, nein; -- aber -- barmherziger Himmel! Was ist das? Frau Bedwin
-- sehen Sie, sehen Sie!»

Und während er hastig die Worte sprach, wies er nach dem Bilde über
Olivers Lehnstuhl und dann auf Oliver selbst hin. Es konnte keine
größere Ähnlichkeit geben; der Knabe war der Dame auf dem Bilde aus den
Augen geschnitten.

Oliver gewahrte die Ursache des plötzlichen Ausrufs seines Wohltäters
nicht; der Schrecken war ihm zu viel gewesen; er war ohnmächtig
geworden. --

Sobald der Baldowerer und Master Bates ihren Zweck erreicht hatten,
alle Aufmerksamkeit von sich ab und auf Oliver zu lenken, schlüpften
sie in eine Seitengasse, um eiligst nach Hause zurückzukehren. Sobald
sie wieder zu Atem gekommen waren, fing Master Bates laut zu lachen
an und rief sich und dem Freunde mit grenzenlosem Vergnügen die
unendlich spaßhafte Szene in das Gedächtnis zurück, wie der geängstete
Oliver gelaufen und überall angerannt war, und wie er selber und der
Baldowerer ihn eifrigst mit gehetzt und das Tuch in der Tasche gehabt
hatten. Sein Freund unterbrach jedoch bald seinen Redefluß und warf das
Bedenken auf, was Fagin dazu sagen würde?

«Was soll er sagen?» meinte Charley.

«Hm!» sagte Jack, pfiff und schnitt sehr bedeutsame Gesichter.

Charley folgte ihm nachdenklich, bald darauf langten sie zu Hause an.
Bei dem Geräusch von Fußtritten auf der krachenden Treppe fuhr der
lustige alte Herr, der vor dem Feuer saß und sich sein Mittagessen
zubereitete, empor. Auf seinem weißen Gesicht lag ein hämisches
Lächeln, als er sich umdrehte und mit einem scharfen Blicke unter
seinen dichten, roten Augenbrauen hervor sein Ohr der Tür zuwandte und
horchte.

«Wie? Was ist das?» murmelte der Jude erschrocken vor sich hin. «Nur
zwei? Wo ist der dritte? Sie werden ihn in dem Gedränge doch nicht
verloren haben? Horch!»

Die Fußtritte kamen näher und näher; endlich öffnete sich die Tür, und
der Baldowerer und Charley Bates traten in das Zimmer.




13. Kapitel.

    Der Leser macht einige neue Bekanntschaften.


«Wo ist Oliver?» fragte der Jude, sich drohend erhebend. «Wo ist der
Junge?»

Die jugendlichen Diebe sahen ihren Lehrmeister erschrocken über dessen
Heftigkeit an und blickten unsicher einander an. Aber sie antworteten
nicht.

«Was ist aus dem Jungen geworden?» fragte der Jude, indem er den
Baldowerer mit festem Griffe beim Kragen packte und fürchterliche
Verwünschungen ausstieß. «Sprich, oder ich erdrossele dich! -- Willst
du sprechen?» fuhr er fort, als keine Antwort erfolgte, und schüttelte
den Baldowerer heftig.

Charley erhob ein jammervolles Geheul, sein Freund riß sich los,
ergriff ein Messer und war im Begriff, es dem Juden in die Seite zu
stoßen, als die Tür geöffnet wurde und ein Vierter, gefolgt von einem
knurrenden, zerbissenen Hunde, eintrat.

«Was gibt's hier, zu allen Teufeln? Spitzbube von Juden, was soll das
bedeuten?»

Die grobe, polternde Stimme gehörte einem vierschrötigen Manne von etwa
fünfundvierzig Jahren mit einem breiten Gesicht und düster grollendem
Blicke an. Sein Bart war seit mehreren Tagen nicht abgenommen und das
eine Auge von einem Schlage angeschwollen, den er erst vor kurzem
erhalten haben mußte. Arm- und Beinschellen dachte man sich bei der
ganzen Erscheinung leicht hinzu.

Er setzte sich gemächlich. «Was sind das hier für Sachen?» fuhr er
fort. «Warum mißhandelst du die Jungen, du alter, unersättlicher Filz
und Pascher?[K] Ich wundere mich nur, daß sie dir die Kehle nicht
abschneiden, was ich unfehlbar tun würde, wenn ich in ihrer Haut
steckte. Ich hätt's längst getan, wenn ich dein Lehrling wäre. Freilich
-- verkaufen hätt' ich deinen Haut- und Knochenkadaver nicht können; du
bist zu nichts gut, denn als ein merkwürdiges Stück von Häßlichkeit in
Spiritus aufbewahrt zu werden, und sie blasen so große Gläser nicht.»

  [K] Hehler.

«Pst, pst! Mr. Sikes,» fiel der zitternde Jude ein, «nicht so laut,
nicht so laut!»

«Ich will dich bemistern; du hast immer Teufeleien im Sinn, wenn du
damit kommst. Du weißt meinen Namen, und ich werd' ihm keine Unehre
machen, wenn die Zeit kommt.»

«Schon gut, schon gut; also Bill Sikes», sagte der Jude kriechend
demütig. «Ihr scheint übler Laune zu sein, Bill.»

Bill überhäufte ihn zur Erwiderung abermals mit Vorwürfen und
Schimpfwörtern und deutete dabei auf so verdächtige Dinge hin, daß
ihn Fagin angstvoll und mit einem Seitenblicke nach den beiden Knaben
fragte, ob er wahnsinnig geworden wäre. Bill machte pantomimisch einen
Knoten unter seinem linken Ohre, wies durch eine Kopfbewegung über
seine rechte Schulter, welche Symbolik der Jude vollkommen zu verstehen
schien, forderte ein Glas Branntwein und fügte die Erinnerung hinzu, es
aber nicht zu vergiften. Er sagte dies scherzend; hätte er jedoch den
satanischen Blick sehen können, mit welchem der Jude sich umwendete, um
nach dem Schranke zu gehen, so würde ihm die Warnung keineswegs unnötig
erschienen sein.

Nachdem er einige Gläser hinuntergestürzt, ließ er sich herab, die
jungen Herren anzureden, was zu einem Gespräch führte, in dessen Laufe
ihm Olivers Gefangennehmung umständlich und mit solchen Ausschmückungen
erzählt wurde, wie sie der Baldowerer für nötig erachtete.

«Ich fürchte, daß er wird etwas lehmern, wodurch wir kommen in
Ungelegenheit», bemerkte der Jude.

«Sehr wahrscheinlich», sagte Bill mit einem boshaften Grinsen. «Du bist
verloren, Fagin.»

Der Jude tat, als ob er die Unterbrechung nicht beachtet hätte, behielt
Sikes scharf im Auge und fuhr fort: «Ich fürchte nur, wenn mir das
Handwerk gelegt würde, möcht's auch noch anderen mehr gelegt werden,
und daß die Geschichte ein schlechteres Ende nimmt für Euch, als für
mich, mein Lieber.»

Sikes fuhr zusammen und blickte den Juden wütend an, der jedoch
die Achseln zuckend gerade vor sich hinstarrte. Nach einem langen
Stillschweigen sagte er mit leiserer Stimme: «Wir müssen zu erfahren
suchen, was sich auf der Polizei zugetragen hat.»

Fagin nickte beifällig.

«Hat er nichts ausgeschwatzt und ist ein Haftbefehl gegen ihn
ausgestellt worden, so ist nichts zu befürchten, bis er wieder
loskommt; dann aber müssen wir seiner so bald wie möglich wieder
habhaft zu werden suchen.»

Der Jude nickte abermals. Der Rat war offenbar gut, nur war die
Ausführung schwierig, da alle vier Gentlemen einen unüberwindlichen
Widerwillen dagegen hegten, einem Polizeiamte nahezukommen. Sie
blickten einander verlegen an, als die beiden jungen Damen eintraten,
deren Bekanntschaft Oliver vor einigen Tagen gemacht hatte. Der Fall
wurde ihnen vorgetragen, und Fagin sprach seine Zuversicht aus, daß
Betsy den Auftrag übernehmen werde. Die junge Dame war zu wohlerzogen
und zu feinfühlend, um einem Mitgliede der Gesellschaft geradezu
oder vielleicht gar mit Schärfe zu widersprechen oder eine Bitte
abzuschlagen. Sie sagte daher keineswegs entschieden nein, sondern
begnügte sich mit der Versicherung, daß sie sich hängen lassen wollte,
wenn sie's täte.

Der Jude wendete sich an ihre Freundin: «Liebe Nancy, was sagst du?»

«Daß ich mich schönstens hüten werde; also gebt Euch nur weiter keine
Mühe, Fagin.»

«Wie soll ich das nehmen?» fiel Sikes grollend ein.

«Just wie ich's gesagt habe, Bill», entgegnete die Dame sehr ruhig.

«Du bist aber eben die rechte Person dazu; es kennt dich hier herum
niemand.»

«Und es tut auch gar nicht not, daß mich jemand kennen lernt, was ganz
gegen meinen Wunsch wäre.»

«Sie geht, Fagin», sagte Sikes.

«Nein, sie läßt's wohl bleiben», eiferte Nancy.

«Ja, ja, sie geht doch», wiederholte Sikes.

Und er hatte recht. Nancy ließ sich endlich durch Geschenke,
Versprechungen und Drohungen bewegen, den Auftrag zu übernehmen. Auch
hatte sie in der Tat weniger als ihre Freundin zu besorgen, mit einem
ihrer zahlreichen Bekannten zusammenzutreffen, da sie erst seit ganz
kurzer Zeit die entlegene, sehr anständige Vorstadt Ratcliffe mit
der Gegend von Fieldlane vertauscht hatte. Der Jude staffierte sie
aus seinen unerschöpflichen Vorräten so aus, wie es dem Zwecke am
angemessensten erschien, und gab ihr einen Korb und einen Hausschlüssel
in die Hand.

«Ach, mein Bruder! mein armer, lieber, kleiner Bruder», begann Nancy
mit überströmenden Tränen und händeringend zu wehklagen. «Ach, was
ist aus meinem Bruder geworden -- wo soll ich ihn finden? O haben Sie
Erbarmen, liebe Herren, und sagen Sie mir, was aus ihm geworden ist!»

Ihre Zuhörer waren entzückt; sie hielt inne, blinzelte lächelnd und
bedeutungsvoll und verschwand.

«Die Nancy ist 'ne gescheite Dirne», sagte der Jude mit feierlichem,
nachdenklichem Kopfnicken zu seinen beiden jungen Freunden, als wenn er
sie mahnen wollte, das eben geschaute glänzende Beispiel nachzuahmen.

«Sie ist 'ne Zierde ihres Geschlechts», stimmte Sikes, sein Glas
füllend und nachdrücklich auf den Tisch schlagend, ein. «Sie lebe hoch,
und möchten ihr alle gleich werden!»

Die Vielgepriesene eilte unterdes nach dem Polizeiamte, wo sie bald,
trotz ein wenig natürlicher Schüchternheit, allein und ohne Beschützer
die Straßen zu durchwandern, glücklich und ohne Gefährde anlangte. Nach
einigen mißlungenen Versuchen wendete sie sich weinend und wehklagend
an den Gefängniswärter, von welchem sie in Erfahrung brachte, daß
Olivers Unschuld ans Licht gekommen und daß er von dem beraubten Herrn
mit fortgenommen worden sei, der in der Gegend von Pentonville wohne,
wohin zu fahren er den Kutscher angewiesen habe. Mit dieser Auskunft
kehrte sie zum Juden zurück.

Sobald sie ihren Bericht erstattet hatte, rief Bill Sikes hastig seinen
Hund, stülpte den Hut auf den Kopf und entfernte sich, ohne sich Zeit
zu der Formalität zu nehmen, der Gesellschaft einen guten Morgen zu
wünschen.

«Wir müssen ihn ausfindig machen; wir müssen wissen, wo er steckt»,
sagte der Jude in großer Aufregung. «Charley, geh auf die Lauer, bis du
etwas von ihm siehst oder hörst. Beste Nancy, ich muß ihn wiederhaben
-- ich verlasse mich ganz auf dich und den Baldowerer. Da, da habt ihr
Geld. Ich entferne mich heut' abend von hier -- ihr wißt, wo ich zu
finden bin. Macht, daß ihr fortkommt -- ihr dürft keinen Augenblick
länger hierbleiben.»

Er stieß alle hinaus, verschloß die Tür hinter ihnen und steckte seine
Kostbarkeiten zu sich. «Er hat nichts ausgeschwatzt auf der Polizei»,
murmelte er; «tut er's aber gegen die Leute, bei denen er sich jetzt
aufhält -- wir werden ihn wiederbekommen und wollen ihm schon stopfen
den Mund.»




14. Kapitel.

    In welchem Mr. Grimwig auftritt.


Oliver erholte sich bald wieder von der Ohnmacht, in die er bei dem
kurzen Ausrufe Mr. Brownlows gefallen war. Der alte Herr und Frau
Bedwin vermieden sorgfältig jedes Gespräch, durch das er wieder an
das Bild oder seine Herkunft und Lage hätte erinnert werden können,
und suchten ihn auf jede Weise angenehm zu unterhalten, ohne ihn
aufzuregen. Als er jedoch am folgenden Tage wieder in das Zimmer der
Haushälterin herunterkam, hob er sogleich die Augen nach der Wand
empor, in der Hoffnung, das Bild der schönen Dame zu erblicken. Er sah
sich getäuscht; es war entfernt worden. Frau Bedwin hatte ihn jedoch
beobachtet.

«Ah!» sagte sie, «es ist nicht mehr da, mein Kind.»

«Ich seh' es, Ma'am!» erwiderte Oliver seufzend. «Warum ist es denn
fortgenommen worden?»

«Weil Mr. Brownlow sagte, es schiene dich unruhig zu machen und könnte
daher deiner Wiederherstellung schaden.»

«Ach, es machte mich gar nicht unruhig, Ma'am. Ich freute mich, es
anzusehen, und hatte es gar zu lieb gewonnen.»

«Nun, nun, mein Kind,» sagte die gute Frau, «es geht dir ja zusehends
besser, und es soll schon wieder aufgehängt werden; ich verspreche es
dir. Laß uns jetzt aber von anderen Dingen sprechen.»

Sie hatte ihm in seiner Krankheit so viel Liebe erwiesen, daß er
sich vornahm, einstweilen nicht mehr an das Bild zu denken. Er hörte
ihr daher aufmerksam zu, als sie begann, ihm von ihren wohlgeratenen
Kindern und ihrem guten, seligen Ehemann zu erzählen. Sodann wurde
Tee getrunken, worauf sie ihn Cribbage spielen lehrte, was er schnell
begriff und eifrig mit ihr spielte, bis es Zeit war, zu Bett zu gehen.

Es folgten nun selige Tage für Oliver. Alles um ihn her war so still,
sauber und ordentlich, und jedermann war so liebevoll gegen ihn, daß
er fast im Himmel zu sein glaubte. Als er imstande war, sich wieder
ordentlich anzukleiden, hatte Mr. Brownlow schon für einen ganz neuen
Anzug gesorgt, und da ihm gesagt wurde, er könnte mit seinen alten
Kleidern tun, was er wollte, so gab er sie der Magd, die sehr gefällig
gegen ihn gewesen war, und sagte ihr, sie möchte sie an einen Juden
verkaufen und das Geld behalten. Die Magd machte sogleich Gebrauch von
der erhaltenen Erlaubnis, Oliver sah durch das Fenster, wie der Jude
seine ganze alte Garderobe zusammenwickelte, einsackte und fortging;
und er freute sich nicht wenig darüber, da er nun nicht mehr zu
fürchten brauchte, die traurigen Lumpen je wieder anlegen zu müssen.

Es mochte etwa eine Woche vergangen sein, als eines Nachmittags Mr.
Brownlow herunterschickte und Oliver zu sich rufen ließ. Frau Bedwin
ordnete eiligst den Anzug und das Haar ihres kleinen Pfleglings und
begleitete ihn selbst bis an Mr. Brownlows Tür. Das Zimmer war mit
Büchern angefüllt, und das einzige Fenster wies in einen kleinen
Blumengarten. Mr. Brownlow legte ein Buch aus der Hand und sagte
Oliver, er möchte näher kommen und sich setzen. Oliver tat, wie
ihm geheißen war, und dachte, wo die Leute wohl gefunden werden
könnten, eine solche Menge von Büchern zu lesen, die geschrieben zu
sein schienen, um die Welt klüger zu machen -- eine Sache, welche
fortwährend erfahreneren Leuten zu schaffen macht, als Oliver Twist es
war.

«Du siehst hier sehr viel Bücher, nicht wahr, mein Kind?» fragte Mr.
Brownlow.

«Ja, sehr viele», erwiderte Oliver; «ich habe noch nie eine solche
Menge von Büchern gesehen.»

«Du sollst sie, wenn du dich gut beträgst, auch lesen, was dir noch
besser gefallen wird als das bloße Beschauen der Bände -- wenn auch
nicht immer; denn es gibt allerdings Bücher, an welchen die Einbände
bisweilen das Beste sind. Möchtest du wohl ein recht gescheiter Mann
werden und selbst Bücher schreiben?»

«Ich möchte lieber in Büchern lesen, Sir», entgegnete Oliver.

«Wie, du möchtest also kein Bücherschreiber sein?» sagte der alte Herr.

Oliver besann sich ein wenig und erwiderte endlich, es bedünke ihn weit
besser, ein Buchhändler zu sein, worüber der alte Herr herzlich lachte,
und wozu er bemerkte, Oliver habe da etwas sehr Gescheites gesagt.
Oliver freute sich über diese Anerkennung, obgleich er durchaus nicht
begriff, wodurch er sie verdient haben möchte.

«Sei nur ohne Furcht», sagte der alte Herr; «ich werde dich nicht zum
Schriftsteller machen, solange es noch ein anderes ehrliches Geschäft
oder Handwerk gibt, das du erlernen kannst.»

«Ich danke, Sir», entgegnete Oliver, und der alte Herr lachte abermals
über den großen Ernst, mit dem er antwortete, und sagte ein paar Worte
von einem merkwürdigen Instinkt, welche Oliver nicht sehr beachtete,
da er sie nicht verstand. Brownlow fuhr darauf in einem womöglich noch
freundlicheren, aber zugleich ernsteren Tone, als er gegen Oliver bis
dahin angenommen, fort: «Sei jetzt recht aufmerksam auf das, was ich
dir sagen werde. Ich denke ohne Rückhalt mit dir zu reden, weil ich
überzeugt bin, daß du mich ebensogut verstehen wirst wie viel ältere
Personen.»

Oliver erschrak. «Ach!» rief er aus, «sagen Sie nicht, daß Sie mich
fortschicken wollen, Sir; weisen Sie mir nicht die Tür, daß ich wieder
auf den Straßen umherirren muß. Lassen Sie mich bei Ihnen bleiben und
Ihnen dienen. Schicken Sie mich nicht in das schreckliche Haus zurück,
woher ich gekommen bin. Erbarmen Sie sich eines armen, verlassenen
Knaben, bester Herr!»

«Mein liebes Kind,» sagte der alte Herr gerührt, «du brauchst nicht
zu fürchten, daß ich meine Hand von dir abziehe, solange du mir keine
Ursache dazu gibst.»

«Das will ich nie, niemals, Sir!»

«Ich hoffe, daß du es nicht tun wirst, glaube es auch nicht. Ich bin
oft getäuscht und betrogen von Leuten, denen ich wohltun wollte, bin
aber trotzdem sehr geneigt, dir zu vertrauen, und ich empfinde eine
größere Teilnahme für dich, als ich sie sogar mir selbst erklären kann.
Die ich am meisten geliebt habe, ruhen längst in ihren Gräbern, und ich
habe auch meines Lebens Glück und Zier begraben -- nicht aber meine
Herzenswärme. Auch herber Kummer hat sie nicht ausgelöscht, sondern
nur noch stärker angefacht; wie denn allerdings Schmerz und Leid unser
Inneres stets reinigen und läutern sollten.» -- Er hatte dies mit
leiser Stimme und mehr vor sich hin als zu Oliver gesprochen, der ganz
still dasaß und kaum zu atmen wagte. -- «Doch ich sagte das nur,» fuhr
der alte Herr wieder heiterer fort, «weil du ein junges Gemüt hast,
und wenn du weißt, daß ich viel gelitten habe, dich vielleicht noch
sorgfältiger hüten wirst, mir abermals wehe zu tun. Du sagst, daß du
eine Waise wärest und ganz allein in der Welt daständest. Alles, was
ich in Erfahrung habe bringen können, bestätigt deine Angaben. Erzähle
mir nun, wer deine Eltern gewesen sind, wo du erzogen und wie du in die
Gesellschaft geraten bist, in welcher ich dich gefunden habe. Sage die
Wahrheit, und wenn ich finde, daß du kein Verbrechen begangen hast, so
soll es dir niemals, solange ich lebe, an einem Freunde fehlen.»

Oliver vermochte vor Schluchzen ein paar Minuten nicht zu antworten,
und als er sich endlich gefaßt hatte und seine Erzählung beginnen
wollte, ließ sich ein Herr zum Tee anmelden.

«Es ist ein Freund von mir, Mr. Grimwig», sagte Brownlow zu Oliver.
«Er hat ein wenig rauhe Manieren, ist aber im Herzen ein sehr wackerer
Mann.»

Oliver fragte, ob er hinuntergehen solle, allein Brownlow hieß ihn
bleiben, und in demselben Augenblick trat Mr. Grimwig, ein korpulenter
alter Herr, gestützt auf einen tüchtigen Handstock -- denn er hatte
ein etwas lahmes Bein --, schon in das Zimmer. Oliver hatte nie ein
so verzwicktes Gesicht gesehen. Grimwig hielt dem Freunde sogleich
auf Armeslänge ein Stückchen Zitronenschale entgegen und polterte,
dergleichen würde ihm überall in den Weg geworfen. «Ich will meinen
eigenen Kopf aufessen, wenn Zitronenschale nicht noch mein Tod ist!»
beteuerte er.

Es war seine gewöhnliche Beteuerung; allein wenn die Erfindung, den
eigenen Kopf zu verspeisen, auch noch gemacht werden sollte, so würde
es einem Herrn, wie Mr. Grimwig war, doch jedenfalls stets sehr
schwerfallen, in einer einzigen Mahlzeit damit zustande zu kommen.

Mr. Grimwig erblickte Oliver, trat ein paar Schritte zurück und fragte
Brownlow verwundert, wer der Knabe wäre.

«Der Oliver Twist, von welchem ich Ihnen erzählt habe», erwiderte
Brownlow.

Oliver verbeugte sich.

«Doch nicht der Knabe, der das Fieber gehabt hat?» sagte Grimwig, sich
noch etwas weiter zurückziehend.

«Gehabt hat», wiederholte Brownlow lächelnd.

Grimwig setzte sich, ohne seinen Handstock zur Seite zu stellen,
beäugelte den hocherrötenden Oliver durch seine Lorgnette und redete
ihn nach einiger Zeit an. «Wie befindest du dich?»

«Danke, Sir, sehr viel besser», erwiderte Oliver.

Brownlow schien zu besorgen, daß sein absonderlicher Freund etwas
Unangenehmes sagen möchte, und hieß Oliver daher hinuntergehen und Frau
Bedwin ankündigen, daß die Herren den Tee erwarteten. Oliver ging mit
Freuden.

«Er ist ein artig aussehender Knabe, nicht wahr?»

«Kann's nicht sagen», entgegnete Grimwig verdrießlich.

«Sie können es nicht sagen?»

«Nein. Ich kann nie einen Unterschied an Knaben entdecken. Ich
kenne nur zwei Arten von Knaben -- Milchsuppengesichter und
Rindfleischgesichter.»

«Zu welcher Art gehört Oliver?»

«Zu den Milchsuppengesichtern. Ich kenne einen Freund, der einen Knaben
mit einem Rindfleischgesicht hat -- einen schönen Knaben, wie ihn seine
Eltern nennen, mit rundem Kopf, roten Wangen und glänzenden Augen --
einen abscheulichen Knaben, wie ich ihn nenne -- mit einem Körper und
Gliedern, die die Nähte seines blauen Anzugs zu sprengen drohen, mit
der Stimme eines Matrosen und einem Wolfshunger. Ich kenne ihn -- den
Bengel!»

«Dann gleicht er Oliver nicht, dem Sie daher nicht zürnen dürfen.»

«Freilich gleicht er dem Oliver nicht, der vielleicht noch schlimmer
ist.»

Brownlow hustete ungeduldig, was seinen Freund höchlich zu ergötzen
schien.

«Ja, ja, er ist vielleicht noch schlimmer», fuhr Grimwig fort. «Woher
stammt er? Wer ist er? Was ist er? Er hat ein Fieber gehabt. Gute
Menschen pflegen keine Fieber zu bekommen, wohl aber schlechte. Ich
habe einen Menschen gekannt, der in Jamaika aufgehängt wurde, weil er
seinen Herrn ermordet hatte. Er hatte sechsmal das Fieber gehabt und
wurde deshalb nicht zur Begnadigung empfohlen.»

Grimwig war im innersten Herzensgrunde sehr geneigt, anzuerkennen, daß
Oliver ein außerordentlich einnehmender Knabe wäre; allein er liebte
noch mehr den Widerspruch, die Zitronenschale hatte ihn gereizt,
er war entschlossen, sich von niemand sein Urteil über einen Knaben
vorschreiben zu lassen, und hatte sich aus diesen triftigen Gründen
von Anfang an vorgenommen, seinem Freunde in allem zu widersprechen.
Als Brownlow daher zugestand, daß seine bisherigen Erkundigungen noch
ungenügend wären, lächelte Grimwig ziemlich boshaft und fragte, ob
die Haushälterin auch wohl regelmäßig das Silbergeschirr nachsähe und
wegschlösse, denn er würde sich eben nicht wundern, wenn sie einmal
einige Löffel oder dergleichen vermißte, usw.

Brownlow, obgleich selbst etwas heftigen Temperaments, ertrug dies
alles sehr gutlaunig, da er die Sonderbarkeiten seines Freundes kannte;
und da sich dieser mit dem Tee und den Semmeln zufrieden zeigte, so
ging alles weit besser, als man hätte erwarten sollen, und Oliver, der
wieder heraufgerufen war, fühlte sich in des sauertöpfischen Herrn
Anwesenheit leichter als zuvor. Als das Teegeschirr hinweggeräumt
wurde, fragte Grimwig, wann sein Freund den Knaben zu veranlassen
gedächte, ihm einen ausführlichen und wahrhaften Bericht über seine
Lebensumstände und Schicksale zu erstatten?

«Morgen früh», erwiderte Brownlow. «Ich wünsche dabei unter vier Augen
mit ihm zu sein. Komm morgen vormittag um zehn Uhr zu mir herauf,
Oliver.»

«Ja, Sir», sagte Oliver. Er antwortete mit einigem Stocken, weil er
dadurch in Verwirrung geraten war, daß Mr. Grimwig ihn bei seiner Frage
so scharf angesehen hatte.

«Ich will Ihnen etwas sagen», flüsterte Grimwig Brownlow in das Ohr;
«er kommt morgen früh nicht herauf zu Ihnen. Ich habe ihn beobachtet.
Er betrügt Sie, lieber Freund.»

«Ich schwöre darauf, daß er's nicht tut», entgegnete Brownlow mit Wärme.

«Ich will meinen Kopf aufessen, wenn er's nicht tut.»

«Und ich bürge mit meinem Leben für seine Wahrhaftigkeit.»

«Und ich mit meinem Kopfe für seine Lügenhaftigkeit.»

«Wir werden sehen», sagte Brownlow, seinen Unwillen bemeisternd.

«Ja, ja, wir werden allerdings sehen», wiederholte Grimwig mit einem
herausfordernden Lächeln.

Das Schicksal wollte es, daß gerade in diesem Augenblick Frau Bedwin
mit einigen Büchern hereintrat, welche Brownlow an demselben Tage von
dem mehrerwähnten Buchhändler gekauft hatte. Sie legte sie auf den
Tisch und schickte sich an, wieder hinauszugehen.

«Lassen Sie den Ladenburschen noch warten», sagte Brownlow; «er soll
etwas mit zurücknehmen -- ein Päckchen Bücher und das Geld für die
gekauften.»

Der Ladenbursche war aber schon wieder fortgegangen.

«Ah, das ist mir aber sehr unangenehm», fuhr Brownlow fort. «Der Mann
braucht sein Geld, und ich würde es auch gern gesehen haben, daß er die
Bücher noch heute zurückerhalten hätte.»

«Schicken Sie sie doch durch Oliver», fiel Grimwig mit einem ironischen
Lächeln ein. «Sie wissen, er wird sie ohne Zweifel richtig abliefern.»

«Ja, lassen Sie sie mich hintragen, Sir», sagte Oliver eifrig. «Ich
will auch den ganzen Weg laufen.»

Brownlow wollte eben erklären, daß er Oliver unter keiner Bedingung
hinschicken werde, als ein boshaftes Husten seines Freundes ihn
bestimmte, seinen Beschluß abzuändern, um Grimwig der Ungerechtigkeit
seines Argwohns zu überführen. Er hieß Oliver die Bücher hintragen
und gab ihm zugleich eine Fünfpfundnote, worauf er zehn Schillinge
zurückbekommen würde.

Oliver versicherte, er würde in zehn Minuten wieder da sein, verbeugte
sich ehrerbietig und eilte hinaus. Frau Bedwin folgte ihm vor die
Haustür, gab ihm ausführliche Anweisungen in betreff des nächsten Weges
und entließ ihn unter vielen wiederholten Ermahnungen, sich nicht zu
überlaufen, sich nicht zu erkälten usf. Es war ihr höchst unangenehm,
ihn aus den Augen lassen zu müssen. Sie hätte auf Mr. Brownlow zürnen
mögen und sah Oliver nach, bis er an der nächsten Ecke angelangt war,
wo er sich noch einmal umwandte und ihr freundlich zunickte.

«Er ist in höchstens zehn Minuten wieder hier», sagte Brownlow und
legte seine Uhr auf den Tisch. «Es wird bis dahin dunkel geworden sein.»

«Sie glauben also wirklich, daß er wiederkommt?»

«Sie nicht?» entgegnete Brownlow lächelnd.

In seinem Freunde regte sich der Widerspruchsgeist gerade mit
besonderer Lebhaftigkeit, und Brownlows Lächeln verstärkte ihn noch.
«Nein!» erwiderte er mit großer Bestimmtheit. «Er steckt in einem
nagelneuen Anzuge, hat ein Paket wertvoller Bücher unter dem Arme und
eine Fünfpfundnote in der Tasche; er wird sich sofort wieder zu seinen
alten Spießgesellen begeben und Sie auslachen. Ich will meinen Kopf
aufessen, wenn er sich jemals wieder hier blicken läßt.»

Er rückte näher an den Tisch, und beide saßen in stummer Erwartung da.
Es ist der Bemerkung wert und wirft ein Licht auf die Bedeutung, welche
wir unseren eigenen Urteilen beilegen, und den Stolz, mit welchem wir
uns auf unsere übereiltesten Schlüsse verlassen, daß Grimwig, obgleich
er kein schlechtes Herz hatte, obgleich es ihn wirklich betrübt
haben würde, wenn er seinen geschätzten Freund betrogen gesehen, im
Augenblick ebenso lebhaft wünschte wie hoffte, Oliver möchte nicht
wiederkommen. Aus solchen Widersprüchen ist die menschliche Natur
zusammengesetzt!

Es wurde so dunkel, daß die Zahlen auf dem Zifferblatt der Uhr
nicht mehr zu erkennen waren; allein die beiden alten Herren saßen
fortwährend da und hefteten schweigend die Blicke auf die Uhr.




15. Kapitel.

    Was Oliver auf dem Wege zum Buchhändler begegnete.


Olivers Rückkehr wurde beiden Herren immer zweifelhafter, zu Grimwigs
Triumph und Brownlows tiefer Betrübnis. Ich hätte nun hier in meinem
Prosaepos die kostbarste Veranlassung, die Leser mit vielen weisen
Betrachtungen über die offenbare Unklugheit zu unterhalten, seinen
Mitmenschen Gutes zu erweisen ohne Aussicht auf irdischen Lohn,
oder vielmehr darüber, wie sehr es die Klugheit erfordere, in einem
besonders hoffnungslosen Falle einige Liebe und Menschenfreundlichkeit
an den Tag zu legen, und sodann dergleichen Schwachheiten für immer
abzulegen. Die Vorteile liegen auf der Hand. Hält sich der, dem ihr
unter die Arme gegriffen, gut und dient ihm euer geleisteter Beistand
zum Wohlergehen, so erhebt er euch bis in den Himmel, ihr werdet
sehr geachtete Leute und gelangt in den Ruf, unendlich viel Gutes im
Verborgenen zu tun, wovon nur der zwanzigste Teil bekannt werde; zeigt
er sich als ein Undankbarer und Nichtswürdiger, so habt ihr euch in die
vortreffliche Stellung gebracht, daß man euch nachsagt, ihr hättet euch
höchst uneigennützig, mildtätig und dienstfertig erwiesen, wäret nur
durch erfahrenen Undank und Verrat menschenfeindlich geworden, und man
könne euch euer Gelübde nicht verdenken, nie wieder einem Menschenkinde
beizuspringen, um nicht durch abermalige Täuschungen verletzt
zu werden. Ich kenne eine Menge Personen, welche die angegebene
Klugheitsregel befolgt haben, und kann versichern, daß sie in der
allgemeinsten und natürlich verdientesten Achtung stehen.

Brownlow gehörte indes zu ihrer Zahl nicht, denn er blieb hartnäckig
dabei, Gutes zu tun um des Guten selbst und um der Herzensberuhigung
und Freude willen, die es ihm gewährte. Täuschungen raubten ihm sein
Vertrauen und seine Milde und seine Menschenfreundlichkeit nicht, und
Undankbarkeit von seiten einzelner führte ihn nicht zu dem Entschlusse,
sich dafür an der ganzen leidenden Menschheit zu rächen. Ich werde
daher die fraglichen vielen weisen Betrachtungen unangestellt lassen,
und sollte dieser Grund ungenügend erscheinen, so kann ich noch
hinzufügen, daß es obendrein gänzlich außer meiner ursprünglichen
Absicht liegt.

Im finsteren Gastzimmer einer kläglichen Winkelschenke, gelegen in der
schmutzigsten Gasse von Little Saffron Hill, saß bei einem Bierkruge
und Branntweinglase ein Mann, in welchem trotz des herrschenden
Halbdunkels kein irgend erfahrener Polizeiagent Bill Sikes verkannt
haben würde. Zu seinen Füßen lag sein weißer, rotäugiger Hund, und
sei es, daß Bill seine Zeit nicht besser anzuwenden wußte, oder daß
er seine üble Laune an irgendeinem Gegenstande auszulassen wünschte,
genug, er versetzte dem Tiere einen derben Fußtritt. Dem Hunde mißfiel
der offenbare Mutwille dieser Behandlung so sehr, daß er nach seines
Herrn Beinen schnappte, Bill ergriff wütend das Schüreisen und sein
Messer, als die Tür sich auftat und der Hund hinausschoß. Zu einem
Streite gehören dem Sprichworte gemäß zwei, und Bill setzte daher den
einmal begonnenen sogleich mit dem Eintretenden fort.

«Verdammter Jude, was trittst du zwischen mich und meinen Hund?» schrie
er ihm entgegen.

«Ich wußt's ja nicht, mein Lieber, wußt's ja nicht, daß Ihr wolltet dem
Hunde zu Leibe», erwiderte Fagin demütig.

«Spitzbube, hast du den Lärm nicht gehört?»

«So wahr mir Gott gnädig ist, nein, Bill, nicht 'nen einzigen Laut.»

«Ja freilich, du hörst nichts, gar nichts», entgegnete Sikes höhnisch;
«ebenso wie du selbst ein und aus schleichst, ohne daß man dich hört.
Ich wollte nur, daß du jetzt der Hund wärst.»

«Warum denn?» fragte Fagin mit einem gezwungenen Lächeln.

«Weil die Regierung, die das Leben solcher Halunken schützt, wie du
einer bist, und die nicht halb so viel Mut haben wie die schlechtesten
Hunde, jedermann erlaubt, seinen Hund abzuschlachten, wenn's ihm
beliebt -- darum!» erwiderte Sikes, sein Messer mit einem sehr
bedeutungsvollen Blicke wieder einsteckend.

Der Jude rieb sich die Hände, setzte sich an den Tisch und zwang sich,
über die Spaßhaftigkeit seines Freundes zu lachen, jedoch war ihm
offenbar dabei nicht besonders wohl zumute.

«Grinse nur, ja grinse nur», sagte Sikes, ihn mit verächtlichem Trotze
anblickend; «über mich sollst du doch nicht lachen, es müßte denn unter
der Nachtmütze sein am Galgen. Ich habe die Hand oben, Fagin, und will
verdammt sein, wenn ich dir den Daumen nicht auf'm Auge halte. Baumele
ich, baumelst du auch; also hüte dich vor mir und trag' hübsch Sorge
für mich.»

«Schon gut, mein Lieber», fiel der Jude ein; «ich weiß das alles;
Gewinn und Gefahr ist gemeinschaftlich bei uns.»

«Hm!» murrte Sikes, als wenn er dächte, der Gewinn möchte wohl zumeist
auf des Juden Seite sein. «Was hast du mir denn aber zu sagen?»

«'s ist alles in den Schmelztiegel gewandert und glücklich wieder
heraus -- da ist Euer Anteil. Ihr erhaltet eigentlich mehr, als Ihr
solltet, mein Lieber; doch da ich weiß, daß Ihr mir schon mal wieder
sein werdet gefällig, und --»

«Haltet ein mit dem Schwätzen», unterbrach ihn Sikes ungeduldig. «Wo
ist's? Her damit!»

«Ja, ja doch, Bill; gönnt mir nur Zeit. Da ist's», versetzte Fagin,
zog ein altes, baumwollenes Taschentuch hervor, knöpfte einen Knoten
auf und reichte Sikes ein Päckchen, der es öffnete und die Goldstücke
hastig zu zählen anfing.

«Ist das alles?» fragte Sikes.

«Ja, alles.»

«Hast du auch das Päckchen nicht aufgemacht auf dem Wege und ein paar
Stück verschluckt? Stell dich nur nicht beleidigt -- hast's ja schon
oft getan. Greif an den Bimbam.»

Fagin klingelte, und es erschien ein anderer Jude, der jünger war, aber
nicht weniger abstoßend und spitzbübisch aussah. Sikes wies stumm nach
dem leeren Kruge hin. Jener verstand den Wink und ging wieder hinaus,
jedoch nicht, ohne Fagin vorher einen Blick zugeworfen zu haben, den
dieser durch ein kaum bemerkbares Kopfschütteln beantwortete. Sikes
hatte sich zufällig gebückt; hätte er den Blick des einen und das
Kopfschütteln des anderen Juden gewahrt, so möchte er der Meinung
gewesen sein, daß ihm diese Pantomimen nichts Gutes bedeuteten.

«Ist niemand hier, Barney?» fragte Fagin den wieder eintretenden Juden.

«Bloß Miß Nancy.»

«Schick sie herein!» sagte Sikes.

Barney blickte Fagin fragend an, ging und kehrte gleich darauf mit
Nancy zurück.

«Du bist auf der Spur, Nancy, nicht wahr, mein Engel?» fragte Bill und
reichte ihr ein gefülltes Glas.

«Ja, Bill», erwiderte die junge Dame, nachdem sie das Glas geleert
hatte; «hab' aber Mühe genug gehabt. Er ist krank gewesen und --»

Nancy bemerkte ein Augenzwinkern Fagins, das eine Warnung vor
übergroßer Mitteilsamkeit zu bedeuten schien. Sie brach ab und fing
an von anderen Gegenständen zu reden. Nach zehn Minuten bekam Fagin
einen Husten, worauf Nancy erklärte, daß es Zeit sei, zu gehen. Sikes
sagte, daß er sie eine Strecke begleiten wolle, da er denselben Weg
habe. Sie entfernten sich daher miteinander. Der Hund folgte in einiger
Entfernung. Fagin sah Sikes durch das Fenster nach, schüttelte die
geballte Faust hinter ihm, murmelte eine grimmige Verwünschung, setzte
sich mit einem schauerlichen Grinsen wieder an den Tisch und war bald
darauf in die Lektüre des Londoner Polizeiblattes vertieft.

Oliver befand sich unterdes auf dem Wege zum Buchhändler, ohne zu
ahnen, daß er dem lustigen alten Juden so nahe wäre. Er geriet in eine
Nebengasse unweit Clerkenwell, bemerkte seinen Irrtum erst, als er sie
bereits über die Hälfte durchwandert hatte, und hielt es für das beste,
um keine Zeit zu verlieren, ihr zu folgen, da sie ihn, wie er meinte,
auch an sein Ziel führen müsse. Er trabte munter vorwärts und dachte
an sein Glück, und was er darum geben würde, wenn er den armen kleinen
Dick daran teilnehmen lassen könnte, als er durch den lauten Ruf: «O
mein lieber kleiner Bruder!» aus seinen Träumereien aufgeschreckt
wurde. Als er aufblickte, umschlossen ihn schon die Arme eines jungen
Mädchens.

«Lassen Sie mich los!» rief Oliver, sich sträubend. «Wer sind Sie? Was
halten Sie mich an?»

Die einzige Antwort darauf war ein Schwall lauter Klagen von seiten des
jungen Mädchens, das einen kleinen Korb und einen Hausschlüssel in der
Hand hatte.

«O gütiger Himmel!» rief das Mädchen aus. «Endlich hab' ich dich
gefunden. Ach, Oliver, o du böser Junge, was hab' ich um deinetwillen
ausgestanden! Gott sei Dank, daß ich dich endlich gefunden habe!»

Das junge Frauenzimmer brach in eine Tränenflut aus und schien so
heftige Krämpfe zu bekommen, daß ein paar mitleidige Frauen einen
dastehenden Fleischerburschen fragten, ob er nicht meinte, daß er zu
einem Doktor laufen müsse, worauf der Fleischerbursche, der eine sehr
große Ruhe, wo nicht ein beträchtliches Phlegma zu besitzen schien,
erwiderte, daß seine Meinung nicht dahin ginge.

«Nein, nein, laßt mich nur», rief jetzt auch das junge Mädchen; «ich
fühle mich schon besser. Und nun komm, mein Junge, geh sogleich mit
mir, mein böser kleiner Liebling.»

«Was gibt's denn?» fragte eine der umstehenden Frauen.

«Ach, er ist vor vier Wochen seinen Eltern entlaufen, guten Leuten, die
sich redlich von ihrer Hände Arbeit nähren, und hat sich unter Gauner
und Landstreicher begeben, daß seine Mutter fast vor Kummer gestorben
wäre.»

«O du kleiner Taugenichts! -- Mach, daß du nach Hause kommst, du
ungeratener Bengel!» riefen die Weiber.

«Ich bin meinen Eltern nicht entlaufen!» rief Oliver in großer Angst.
«Ich habe weder Schwester noch Eltern. Ich bin eine Waise und wohne in
Pentonville.»

«Ach du gütiger Himmel, wie trotzig er schon geworden ist!» schluchzte
das junge Mädchen.

«Ei, Nancy!» rief Oliver, der jetzt erst ihr Gesicht sah, im höchsten
Erstaunen aus.

«Sie sehen, er kennt mich», sagte Nancy. «Helfen Sie mir ihn nach Hause
bringen, liebe Leute; seine Eltern und wir alle sterben sonst noch vor
Kummer über ihn.»

«Zu allen Teufeln, was ist das hier?» schrie ein aus einem Bierladen
hervorstürzender Mann. «Oliver, Satansbrut, komm augenblicklich mit
nach Hause zu deiner armen Mutter. Sofort kommst du mit!»

«Ich gehöre nicht zu ihnen. Ich kenne sie nicht, Hilfe, Hilfe!» rief
Oliver, indem er sich unter dem festen Griff des Mannes verzweifelt
wand.

«Hilfe!» polterte Sikes. «Ich will dir gleich helfen. Was sind das für
Bücher? -- Ohne Zweifel gestohlen -- her damit!»

Er entriß ihm das Päckchen und versetzte ihm damit einen heftigen
Schlag auf den Kopf.

«So ist's recht; das wird ihn schon wieder zur Besinnung bringen!»
riefen die Weiber.

«Sollt's auch meinen», rief der Mann, gab Oliver noch ein paar Schläge
auf den Kopf und packte ihn beim Kragen. «Komm, du kleiner Taugenichts!
Hier, Tyras, paß auf ihn auf! Paß auf!»

Noch geschwächt von seiner Krankheit, betäubt durch die Schläge und
das Überraschende des ganzen Vorganges, in Schrecken gesetzt durch
das Knurren des Hundes und die Brutalität des baumstarken Mannes, und
überwältigt durch den Beifall, den die Umstehenden seinen Angreifern
gaben -- was konnte das geängstete Kind tun? Es war dunkel geworden,
die Gasse sah an sich selbst schon verdächtig aus, Hilfe war nirgends
zu erblicken, Widerstand nutzlos. Ohne recht zu wissen, wie ihm
geschah, fühlte sich Oliver durch ein Labyrinth von engen Straßen
geschleppt, und sein jeweiliges Rufen verhallte um so mehr, da er so
schnell fortgerissen wurde, daß er keinen Augenblick zu Atem kommen
konnte; doch würde es auch von niemand beachtet worden sein.

       *       *       *       *       *

Die Gaslampen waren angezündet; Frau Bedwin erwartete mit herzpochender
Ungeduld, daß die Haustür sich auftun sollte; die Magd war zwanzigmal
die Straße hinuntergelaufen, um nach Oliver auszusehen; die beiden
alten Herren saßen beharrlich im Dunkeln neben der zwischen ihnen
liegenden Uhr.




16. Kapitel.

    Was sich mit dem entführten Oliver begab.


Die engen Straßen und Gäßchen mündeten endlich auf einen weiten,
offenen Platz, um den rings Stallungen standen zum Zeichen, daß
hier ein Viehmarkt war. Sikes verlangsamte seinen Schritt, als sie
diese Gegend erreichten, da das Mädchen völlig außerstande war, den
Laufschritt, den sie bisher angeschlagen hatten, länger auszuhalten.
Sich an Oliver wendend, befahl er ihm barsch, Nancys Hand zu fassen.

«Hörst du nicht?» brummte Sikes, als Oliver zögerte und sich umsah.

Sie befanden sich in einem finsteren, ganz abgelegenen Stadtteil, und
Oliver sah nur zu gut ein, daß Widerstand nutzlos war. Er streckte
seine Hand aus, die Nancy fest mit der ihrigen umklammerte.

Der Abend war dunkel und feucht; die Lichter in den Läden konnten kaum
gegen den Nebel ankämpfen, der immer dichter wurde und die Straßen und
Häuser in ein undurchdringliches Grau hüllte. Sie hatten Smithfield
erreicht, als tiefe Glockenschläge die Stunde verkündeten. Sikes und
Nancy standen bei den ersten Schlägen still und wandten sich nach der
Richtung um, aus welcher die Töne erschallten.

«Acht Uhr, Bill», sagte Nancy, als die Glocke aufhörte zu schlagen.

«Ich habe selbst Ohren», erwiderte Sikes mürrisch.

«Ich möchte wohl wissen, ob *sie* es schlagen hören können?» fuhr Nancy
fort.

«Natürlich können sie's», sagte Sikes. «Es war um Bartholomäi, als
ich in Dobes[L] gesteckt wurde, und auf dem ganzen Markt schnarrte
keine Pfennigtrompete, die ich nicht gehört hätte. Nachdem ich für die
Nacht eingeschlossen war, machte der Lärm und das Getöse draußen das
vermaledeite alte Gefängnis so still und einsam, daß ich mir den Kopf
hätte einrennen mögen an den Basteln[M].»

  [L] Gefängnis.

  [M] Eisenstäbe.

«Die armen Kerls! Ach, Bill, was sie für schmucke junge Leute sind!»

«Ja, ja, so sprecht ihr Weibsbilder alle!» erwiderte Sikes in einem
Anfluge von Eifersucht. «Schmucke junge Leute! Doch sie sind so gut wie
tot, also mag's gleichviel sein.»

Er faßte den Knaben wieder fester und trieb zur Eile an.

«Noch einen Augenblick», sagte das Mädchen; «ich würde nicht
vorbeilaufen, wenn Ihr's wär't, der zum Galgen herausgeführt würde,
wenn's wieder acht schlägt. Ich würde auf und nieder travallen, bis ich
niedersänke, und wenn fußhoher Schnee läge, und ich hätte kein warmes
Tuch, mich einzuhüllen.»

«Das sollte mir wohl viel helfen», bemerkte der nichtsentimentale
Sikes. «Könnt'st du mir nicht ä Kulm[N] und ä zwanzig Ellen Kabot[O]
'neinpraktizieren, so möcht'st du fünfzig Meilen laufen oder ganz zu
Hause bleiben, es wäre mir alles nichts nütze. Vorwärts, steh' hier
nicht länger und paternelle[P] nicht!»

  [N] Feile.

  [O] Seil, Strick.

  [P] mach keine Predigten.

Das Mädchen brach in ein Gelächter aus, ergriff Olivers Hand, und sie
eilten weiter. Oliver fühlte, daß ihre Finger zitterten, und als sie an
einer Gaslampe vorüberkamen, sah er, daß ihr Gesicht totenblaß war.

Sie lenkten nach einer halben Stunde in eine enge, schmutzige Gasse
ein, die fast ganz von Trödlern bewohnt zu sein schien, und standen
vor einem verschlossenen Laden still. Das Haus schien unbewohnt zu sein
und sah halb verfallen aus. Über der Tür war eine Tafel angenagelt, auf
welcher zu lesen war, daß das Haus zu vermieten sei; sie schien jedoch
dort schon jahrelang befestigt gewesen zu sein.

Nancy bückte sich, und Oliver hörte den Ton einer Glocke. Sie gingen
auf die entgegengesetzte Seite der Straße und stellten sich unter eine
Laterne. Ein Geräusch ließ sich hören, als ob ein Fenster vorsichtig in
die Höhe geschoben würde, und gleich darauf öffnete sich geräuschlos
die Tür. Mr. Sikes packte den erschrockenen Knaben jetzt ohne Umstände
beim Kragen, und im nächsten Augenblick befanden sich alle drei im
Innern des Hauses. Hier war es stockfinster. Sie warteten, bis die
Person, die sie eingelassen hatte, die Tür wieder verschlossen und mit
einer Sicherheitskette verwahrt hatte.

«Ist jemand hier?» fragte Sikes.

«Nein!» erwiderte eine Stimme, die Oliver bekannt vorkam.

«Ist der Alte hier?» fragte der Dieb.

«Ja,» antwortete die Stimme, «und er wird sicher sehr erfreut sein, Sie
zu sehen.»

«Machen Sie Licht,» versetzte Sikes, «oder wir brechen uns den Hals
oder treten auf den Hund. Nehmen Sie Ihre Beine in acht, wenn Sie es
tun.»

«Bleiben Sie einen Augenblick stehen; ich werde Licht bringen»,
erwiderte die Stimme. Man hörte, wie sich der Sprecher entfernte, und
eine Minute später erschien die Gestalt John Dawkins', genannt der
«gepfefferte Baldowerer». Der junge Herr gab nur durch ein spöttisches
Grinsen kund, daß er Oliver wiedererkannt habe, und bat die Besucher,
ihm eine Anzahl Stufen hinunter zu folgen. Sie gingen durch eine
leere Küche und traten in ein niedriges, dumpfiges Gemach ein. Ein
lautes Gelächter schallte ihnen entgegen. Charley Bates wälzte sich
im eigentlichen Sinne vor Vergnügen über den gar zu kostbaren Spaß
auf dem Boden, riß sodann Jack Dawkins das Licht aus der Hand, hielt
es Oliver dicht vor das Gesicht und beschaute ihn von allen Seiten,
während ihm Fagin scherzhafterweise tiefe Verbeugungen machte und der
Baldowerer, der von ernsterem Wesen war und sich nicht leicht der
Heiterkeit überließ, wenn es Geschäfte zu verrichten galt, sorgfältig
seine Taschen durchsuchte.

«Ich freue mich unendlich, Sie so wohl zu sehen, mein Lieber», sagte
der Jude. «Der Gepfefferte soll Ihnen geben einen anderen Anzug, damit
Sie den sonntäglichen nicht verderben gleich. Warum schrieben Sie's
nicht, daß Sie kommen wollten -- wir hätten dann treffen können noch
bessere Vorbereitungen -- aber Sie sollen dennoch etwas Warmes bekommen
zum Abendbrot.»

Jetzt lächelte sogar der Baldowerer; da er jedoch in diesem Augenblicke
die Fünfpfundnote hervorzog, so ist es zweifelhaft, ob der Witz Fagins
oder die erfreuliche Entdeckung seine Heiterkeit erregte.

«Holla, was ist das?» rief Sikes und trat auf den Juden zu, als
derselbe die Banknote hinnahm. «Diese ist mein, Fagin!»

«Nein, nein, mein Lieber», entgegnete der Jude. «Mein, Bill, mein; Ihr
sollt die Bücher haben.»

«Bekomm' ich und Nancy sie nicht,» sagte Sikes, mit entschlossener
Miene den Hut aufsetzend, «so bring' ich den Buben wieder zurück.»

Der Jude fuhr empor und Oliver gleichfalls, obgleich aus einem ganz
anderen Grunde; er hoffte, der Streit würde damit enden, daß man ihn
wieder nach Pentonville zurückbrächte. Allein Sikes entriß dem Juden
unter Schelten und Drohen die Banknote, faltete sie kaltblütig zusammen
und knüpfte sie in den Zipfel seines Halstuchs.

«'s ist für unsere Mühe und noch nicht halb genug», sagte er. «Behaltet
Ihr die Bücher, wenn Ihr gern lest, und wo nicht, schlaget sie los!»

«Es sind prächtige Bücher; nicht wahr, Oliver?» fiel Charley Bates ein,
als er die klägliche Miene gewahrte, mit der Oliver zu seinen Peinigern
emporblickte.

«Sie gehören dem alten Herrn», sagte Oliver händeringend, «dem lieben,
guten alten Herrn, der mich in sein Haus nahm und mich pflegen ließ,
als ich todkrank lag. O bitte, schicken Sie sie zurück, schicken Sie
ihm die Bücher und das Geld zurück! Behalten Sie mich hier mein Leben
lang, aber bitte, bitte, schicken Sie sie nur zurück. Er wird glauben,
daß ich sie gestohlen hätte -- und die alte Dame und alle, die so
freundlich gegen mich waren werden es denken. O haben Sie Erbarmen und
schicken Sie die Bücher und das Geld zurück!»

Oliver fiel vor dem Juden auf die Knie nieder und hob flehend und ganz
in Verzweiflung die Hände zu ihm empor.

«Der Bube hat recht», sagte Fagin, listig umherblickend und die
buschigen Augenbrauen zusammenkneifend. «Du hast recht, Oliver, hast
ganz recht; sie werden allerdings glauben, daß du sie gestohlen hast.
Ha, ha, ha!» kicherte er und rieb sich die Hände; «es hätte sich ganz
unmöglich treffen können besser, und wenn wir noch so gut gewählt
hätten die Zeit.»

«Versteht sich», fiel Sikes ein; «ich wußt's gleich im selbigen
Augenblick, als ich ihn durch Clerkenwell mit den Büchern unterm
Arm daherkommen sah. 's ist nun alles gut. Es müssen schwachköpfige
Betbrüder sein -- hätten ihn sonst gar nicht zu sich genommen; und sie
werden auch keine Nachfrage anstellen, aus Furcht, daß sie ihn anklagen
müßten und ihn gerumpelt[Q] zu sehen. Wir haben ihn jetzt fest genug.»

  [Q] Auf den Schub bringen -- deportieren.

Oliver hatte unterdes bald Sikes, bald Fagin angesehen, als wenn er
ganz betäubt wäre und kaum verstände, was gesprochen wurde; allein
bei Bills letzten Worten sprang er plötzlich empor und stürzte unter
einem Geschrei nach Hilfe aus der Tür hinaus, daß die nackten Wände des
Hauses davon widerhallten.

«Halt den Hund zurück, Bill,» schrie Nancy, eilte vor die Tür und
verschloß sie, als der Jude mit seinen beiden Zöglingen Oliver
nachgestürzt war; «halt den Hund zurück; er reißt ihn in Stücke!»

«Ist ihm gerade recht!» rief Sikes und suchte sich von dem Mädchen
loszumachen. «Laß mich los, oder ich renn dir den Kopf gegen die Wand!»

«Ist mir alles gleichviel, Bill, ist mir alles gleichviel», schrie das
Mädchen, sich heftig gegen ihn sträubend; «er soll nicht von dem Hunde
zerrissen werden, und wenn es mein Tod ist!»

«So!» tobte Sikes; «sollst nicht lange warten auf deinen Tod, wenn du
nicht im Augenblick ablässest!»

Er schleuderte sie in die fernste Ecke des Gemachs, gerade als der
Jude, Jack und Charley den Flüchtling wieder hereinschleppten.

«Was gibt's hier?» fragte Fagin.

«Ich glaube, die Dirne ist toll geworden», erwiderte Sikes in Wut.

«Nein, ich bin nicht toll», rief Nancy blaß und atemlos dazwischen;
«nein, Fagin, glaubt's nicht!»

«Dann sei ruhig -- willst du wohl?» sagte der Jude mit drohender
Gebärde.

«Das will ich auch nicht!» erwiderte Nancy mehr schreiend als redend.
«Was willst du nun?»

Mr. Fagin war mit den Sitten und Gebräuchen der Spezies von
Menschenkindern hinlänglich bekannt, welcher Miß Nancy angehörte, um
sich ziemlich überzeugt zu fühlen, daß es einigermaßen gefährlich sein
würde, die Unterhaltung mit ihr für den Augenblick fortzusetzen. Er
wendete sich daher, um die Aufmerksamkeit der Gesellschaft abzulenken,
zu Oliver.

«Du wolltest also fortlaufen, mein Lieber?» sagte er, einen Knotenstock
aufhebend, der am Kamine lag; «wolltest rufen die Polizei -- nicht
wahr, mein Schatz? Ich will dich von der Krankheit kurieren, lieber
Engel!»

Er hatte bei diesen Worten Oliver beim Arme gefaßt, versetzte ihm einen
Schlag über den Rücken und hob den Knotenstock wieder empor, als Nancy
auf ihn zustürzte, ihm den Stock aus der Hand riß und in das Feuer
schleuderte.

«Ich leid's nimmermehr, Fagin!» schrie sie. «Ihr habt den Knaben, und
was wollt Ihr mehr? Laßt ihn -- laßt ihn zufrieden, oder ich tue etwas
an Euch, das mich vor meiner Zeit an den Galgen bringt!» Sie stampfte
bei dieser Drohung heftig mit den Füßen und blickte mit verbissenen
Lippen, geballten Fäusten und blaß vor Zorn und Wut abwechselnd den
Juden und Sikes an.

«Ah, Nancy!» sagte der Jude nach einer kurzen, verlegenen Pause
beschwichtigend; «du -- du übertriffst dich wirklich heute abend selbst
-- ha, ha, ha! -- spielst ganz prachtvoll deine Rolle, liebes Kind!»

«So!» entgegnete Nancy; «nehmt Euch nur in acht, daß ich sie nicht zu
gut für Euch spiele. Ich sage es Euch vorher, Ihr werdet Euch sehr
schlecht dabei stehen!»

Es gibt wenige Männer, die sich nicht gern enthielten, ein in Wut
geratenes und obendrein von nichtsachtender Verzweiflung beseeltes
Frauenzimmer noch mehr zu reizen. Der Jude sah ein, daß es ihm nichts
helfen könne, sich noch länger zu stellen, als wenn er Nancys Zorn
für bloß erkünstelt hielte, fuhr unwillkürlich einige Schritte zurück
und blickte halb zitternd, halb verzagend nach Sikes. Dieser mochte
glauben, sein persönliches Ansehen fordere es, Nancy baldigst wieder
zur Vernunft zu bringen, und begann daher seine Operationen mit
zahlreichen und kräftigen Drohungen und Verwünschungen, wobei er den
Beweis lieferte, daß er es in diesem Genre in der Tat zur Meisterschaft
gebracht hatte. Als sie keinen sichtbaren Eindruck machten, ging er zu
noch überzeugenderen Argumenten über. «Was soll das bedeuten, Dirne?»
tobte er unter Hinzufügung einer Verwünschung, die die Blindheit so
gewöhnlich als die Masern machen würde, wenn der Himmel sie nur halb
so oft wahr machte, als man sie auf Erden hört. «Was willst du damit
bezwecken? Weißt du, zum Geier, wer du bist -- was du bist?»

«O ja, ja; ich weiß es nur zu gut!» erwiderte Nancy unter krampfhaftem
Lachen, dabei den Kopf hin und her wiegend, um gleichgültig zu
erscheinen, was ihr jedoch schlecht gelang.

«Dann sei ruhig, oder ich werde dich auf 'ne lange Zeit zum
Stillschweigen bringen.»

Sie lachte abermals, blickte flüchtig nach Sikes, wendete das Gesicht
ab und biß sich die Lippen blutig.

«Du bist mir die Rechte, dich auf die menschenfreundliche und honette
Seite zu legen!» fuhr er verächtlich fort. «Der Bursch würde 'ne
saubere Freundin an dir haben, wozu du dich aufwirfst!»

«Und beim allmächtigen Gott, ich bin es!» rief sie mit
leidenschaftlicher Heftigkeit; «und ich wollte lieber, daß ich auf der
Straße tot niedergefallen oder in das Gefängnis geworfen wäre, statt
derer, denen wir so nahe waren, als daß ich mich dazu hergegeben hätte,
ihn hierher zu bringen. Er ist von heute abend an ein Dieb, ein Lügner,
ein Mörder, ein Teufel und alles, was nur schlecht und verworfen heißen
mag; -- ist das nicht genug für den alten Halunken -- muß er ihn
obendrein schlagen?»

«Hört, Bill,» fiel der Jude dringend und nach dem mit gespanntem
Ohr zuhörenden Knaben hindeutend ein, «wir müssen freundliche Worte
gebrauchen, freundliche Worte, Bill.»

«Freundliche Worte!» schrie das in seiner Wut schrecklich aussehende
Mädchen; «freundliche Worte, Ihr Schuft! Ja, die verdient Ihr auch von
mir! Ich habe gestohlen für Euch, als ich noch nicht halb so alt war
wie dies Kind hier, und bin in demselben Geschäft und demselben Dienst
seit zwölf Jahren gewesen; wißt Ihr das nicht? Sprecht, wißt Ihr es
nicht?»

«Ja, ja doch», erwiderte der Jude besänftigend; «du hast ja aber auch
davon dein Brot.»

«Freilich, ich habe mein Bettelbrot davon,» schrie sie immer heftiger,
«und die kalten, nassen, schmutzigen Straßen sind meine Wohnung; und
Ihr seid der ruchlose Mann, der mich Tag und Nacht hinaustreibt und
mich Tag und Nacht hinaustreiben wird, bis ich im Grabe liege.»

«Ich füge dir ein Leid zu,» versetzte der Jude, durch diese Vorwürfe
gereizt, «ein Leid, das schlimmer ist als das, von dem du sprichst,
wenn du noch ein Wort sagst.»

Nancy erwiderte nichts mehr, zerraufte aber in einem Übermaß von
Leidenschaft ihr Haar, stürzte auf Fagin zu, und auf seinem Gesichte
würden ohne Zweifel sichtbare Spuren ihrer Rache zurückgeblieben sein,
hätte nicht Sikes eben noch zur rechten Zeit ihre Arme festgehalten.
Sie bemühte sich vergeblich, sich von ihm loszureißen, und sank in
Ohnmacht. «Es ist nun alles wieder in Ordnung», bemerkte Sikes, sie in
eine Ecke tragend. «Sie besitzt außerordentliche Körperkräfte, wenn sie
sich in diesem Zustand befindet.» Der Jude wischte sich die Stirn und
lächelte, und sowohl er wie Sikes und die Knaben schienen den ganzen
Vorfall als einen gewöhnlichen, im Geschäft häufig vorkommenden zu
betrachten.

«Es ist doch das Schlimmste, mit Weibern zu tun zu haben», bemerkte
der Jude, den Stock wieder beiseite stellend; «aber sie sind schlauer
als wir, und wir können ohne sie nicht fertig werden. Charley, bringe
Oliver zu Bett.»

«Nicht wahr, Fagin, er soll morgen seine besten Kleider nicht tragen?»
fragte Charley Bates grinsend, und der Jude verneinte, Charleys
liebliches Grinsen erwidernd. Master Bates schien sich seines Auftrages
höchlich zu freuen, führte Oliver in das anstoßende Gemach, in welchem
einige Betten der Art standen, wie er sie bereits kennen gelernt, und
zog mit unbezwinglichem Gelächter die alten Kleidungsstücke hervor, die
sich Oliver so gefreut hatte, ablegen zu dürfen, und die Fagin auf die
erste Spur seines Aufenthaltes bei Mr. Brownlow gebracht hatten.

«Zieh' die Sonntägischen aus,» sagte Charley, «ich will sie Fagin zum
Aufheben geben. Welch' ein prächtiger Spaß!»

Der arme Oliver gehorchte widerstrebend und wurde darauf von Charley im
Dunkeln gelassen und eingeschlossen. Master Bates' Gelächter und die
Stimme Betsys, die nach einiger Zeit erschien, und ihre Freundin zum
Bewußtsein zurückzurufen sich bemühte, wären gar wohl geeignet gewesen,
ihn unter anderen Umständen wach zu erhalten; allein er war erschöpft
und unwohl und schlief daher bald ein.




17. Kapitel.

    Olivers Schicksal bleibt fortwährend günstig.


In jedem guten Melodrama, in dem viel von Hauen und Stechen die Rede
ist, wechseln auf der Bühne komische und tragische Szenen so regelmäßig
wie die roten und weißen Lagen eines Stücks durchwachsenen Specks.
Diese Abwechselungen erscheinen uns abgeschmackt, sind indes keineswegs
unnatürlich. Die Übergänge im wirklichen Leben von wohlbesetzten
Tischen zu Sterbebetten oder von Trauer- zu Festtagskleidern sind nicht
minder schroff oder gefühlverletzend -- wir aber sind beschäftigte
Mitspielende statt bloßer Zuschauer, was einen unermeßlichen
Unterschied bildet; den Schauspielern sind die plötzlichen Übergänge
nicht auffällig, sie haben sozusagen keine Augen für dieselben, die
von den Zuschauern verkehrt, unnatürlich, extravagant genannt werden.
Verdamme mich daher nicht zu voreilig, geneigter Leser, wenn du in
meinem Buche einen häufigen Wechsel des Schauplatzes und der Szenen
findest, sondern erzeige mir die Gunst, zu prüfen, ob ich recht oder
unrecht dabei gehabt habe. Meine Erzählung soll meiner Absicht nach
wahr sein und ohne unnötige Abschweifungen auf ihr Ziel lossteuern. Ich
bitte, folge mir für jetzt vertrauensvoll nach der Stadt, in welcher
mein kleiner Held das Licht der Welt erblickte.

Mr. Bumble trat eines Morgens früh aus dem Armenhause mit der
wichtigsten Miene heraus, und durchschritt die Straßen mit einer
Haltung und einem Wesen, daß man es ihm sogleich ansah, sein Inneres
war von Gedanken erfüllt, zu groß, um sie aussprechen zu können. Er
hielt sich nicht unterwegs auf, um sich mit den kleinen Krämern und
anderen, die ihn anredeten, in herablassender Weise zu unterhalten,
sondern erwiderte ihre Begrüßungen nur mit einer hoheitsvollen
Handbewegung, und hemmte seinen würdevollen Schritt erst, als er vor
der Anstalt stand, in der Mrs. Mann die Armenkinder mit parochialer
Sorgfalt pflegte.

«Dieser verwünschte Kirchspieldiener!» sagte Mrs. Mann zu sich selbst,
als sie das bekannte Rütteln an der Pforte hörte. «Ob er nicht schon in
aller Herrgottsfrühe herauskommt! Schau, Mr. Bumble, soeben habe ich an
Sie gedacht. Ja, verehrter Herr, es ist mir ein wirkliches Vergnügen,
Sie wieder einmal zu sehen! Treten Sie, bitte, näher.»

Der erste Satz war zu Susanne gesprochen worden, die Freudenbezeigungen
dagegen zu Mr. Bumble, als die gute Dame die Gartenpforte öffnete und
ihn mit großer Höflichkeit und Ehrerbietung ins Haus nötigte.

«Mrs. Mann,» sagte Mr. Bumble, indem er sich mit großer Feierlichkeit
und Würde auf einen Stuhl niederließ, «Mrs. Mann, Ma'am, ich biete
Ihnen einen guten Morgen.»

«Ich danke Ihnen und biete Ihnen auch meinerseits einen guten Morgen»,
erwiderte Mrs. Mann freundlich lächelnd; «ich hoffe, Sie befinden sich
wohl, Sir.»

«So -- so, Mrs. Mann,» antwortete der Kirchspieldiener; «man ist in der
Parochie nicht immer auf Rosen gebettet.»

«Ach ja, das ist man in der Tat nicht», versetzte die Dame, und alle
Armenkinder würden ihr laut beigepflichtet haben, falls sie ihre Worte
gehört hätten.

«Ein Leben im Dienste der Parochie», fuhr Mr. Bumble fort, «ist ein
Leben voller Mühseligkeiten und Plagen, Ma'am; aber alle öffentlichen
Charaktere, darf ich wohl sagen, müssen unter Verfolgungen leiden.»

Mrs. Mann, die nicht genau wußte, was der Kirchspieldiener meinte,
erhob ihre Hände mit einem Seufzer des Einverständnisses.

«Ach ja,» bemerkte Mr. Bumble, «Sie haben wohl ein Recht zu seufzen,
Ma'am.»

Da Mrs. Mann fand, sie habe richtig gehandelt, seufzte sie von neuem,
offenbar zur Befriedigung des «öffentlichen Charakters»; denn Mr.
Bumble sagte, ein wohlgefälliges Lächeln unterdrückend: «Mrs. Mann, ich
gehe nach London.»

«Was Sie sagen, Mr. Bumble!» erwiderte Mrs. Mann erstaunt.

«Nach London, Ma'am,» wiederholte der Kirchspieldiener
unerschütterlich, «und zwar in einer Postkutsche. Ich und zwei Arme,
Mrs. Mann.»

«Sie benutzen eine Postkutsche, Sir?» fragte Mrs. Mann. «Ich glaubte,
es wäre immer üblich, Arme auf offenen Karren zu verschicken.»

«Das geschieht, wenn sie krank sind», entgegnete der Kirchspieldiener;
«wir setzen die kranken Armen bei Regenwetter auf offene Karren, damit
sie sich nicht erkälten. Die Postkutsche nimmt diese beiden außerdem
um ein sehr Billiges mit, und wir finden, es kommt uns um zwei Pfund
wohlfeiler zu stehen, wenn wir sie in ein anderes Kirchspiel schaffen
können, als wenn wir sie hier begraben müssen. Hahaha! -- Aber wir
vergessen das Geschäft, Ma'am,» fuhr er ernst werdend fort; «hier ist
das Kostgeld für den Monat.»

Mr. Bumble holte eine kleine Rolle mit Silbergeld aus seiner
Brieftasche hervor und bat um eine Quittung; Mrs. Mann schrieb sie
sofort.

«Ich danke Ihnen recht sehr, Mr. Bumble; ich bin Ihnen in der Tat für
Ihre Liebenswürdigkeit sehr verbunden.»

Mr. Bumble nickte gnädig in Anerkennung der Höflichkeit Mrs. Manns, und
erkundigte sich sodann nach dem Befinden der Kinder.

«Gott segne ihre lieben kleinen Herzchen», erwiderte Mrs. Mann bewegt;
«sie befinden sich den Umständen angemessen wohl, die lieben Kleinen!
Natürlich bis auf die zwei, die vergangene Woche gestorben sind, und
den kleinen Dick.»

«Geht es mit dem Jungen immer noch nicht besser?» fragte Mr. Bumble.
Mrs. Mann schüttelte den Kopf.

«Er ist ein schlecht beanlagtes, lasterhaftes Parochialkind mit üblen
Angewohnheiten», sagte Mr. Bumble ärgerlich. «Wo ist er?»

«Ich werde ihn Ihnen in einer Minute herbringen, Sir», gab Mrs. Mann
zur Antwort.

Nach einigem Suchen wurde Dick entdeckt und nach gründlicher Säuberung
unter der Pumpe Mr. Bumble vorgeführt.

Das Kind war bleich und mager; seine Wangen waren eingesunken und seine
Augen groß und fieberisch glänzend. Die armselige Parochialkleidung,
die Livree seines Elends, hing schlotternd um seinen schwächlichen
Körper, und seine jungen Glieder waren welk wie die eines alten Mannes.

«Wie geht es dir?» fragte Mr. Bumble den Knaben, der zitternd dastand
und seine Augen nicht vom Fußboden zu erheben vermochte.

«Ich glaube, daß ich bald sterben muß,» erwiderte der kleine Patient,
«und ich freue mich auch recht darauf, denn ich habe ja keine Freude
hier. Sagen Sie doch Oliver Twist, wenn ich erst tot bin, ich hätte
ihn sehr lieb gehabt und tausendmal an ihn gedacht, wie er allein und
hilflos umherwandern müßte --»

Er hatte die Worte mit einer Art von Verzweiflung gesprochen, ohne
sich durch Frau Manns pantomimische Drohungen irren zu lassen; doch
erstickten endlich Tränen seine Stimme.

«Frau Mann,» bemerkte Bumble, «ich sehe wohl, der eine ist wie der
andere. Sie sind samt und sonders durch den Taugenichts Oliver Twist
verführt und verdorben worden. Ich werde dem Direktorium Anzeige von
dem Falle machen, damit strengere Maßregeln angeordnet werden. Lassen
Sie ihn sogleich wieder hinausbringen!»

Dick wurde in den Kohlenkeller gebracht, und Bumble begab sich wieder
zur Stadt zurück, wo er sich in kürzester Frist reisefertig machte
und mit den beiden nach London zu schaffenden Armen die bestellten
Außenplätze der Postkutsche einnahm. Die beiden Armen klagten viel
über Kälte; Bumble hüllte sich dicht in seinen Mantel, philosophierte
ziemlich mißvergnügt über den Undank und die unablässigen unzufriedenen
Klagen der Menschen und fühlte sich erst wieder recht behaglich, als er
in dem Gasthause, in welchem die Kutsche anhielt, sein gutes Abendessen
eingenommen, seinen Stuhl an den Kamin gestellt hatte, sich niederließ
und ein Zeitungsblatt zur Hand nahm. Wer beschreibt sein Erstaunen, als
er gleich darauf nachstehenden Artikel fand:

«*Fünf Guineen Belohnung*.

Am vergangenen Donnerstag abend hat sich ein Knabe namens Oliver
Twist aus seiner Wohnung in Pentonville entfernt und mit oder ohne
seine Schuld nichts wieder von sich hören lassen. Es werden hierdurch
demjenigen fünf Guineen geboten, der eine Mitteilung zu machen geneigt
und imstande ist, die zur Wiederauffindung des besagten Oliver Twist
führen kann, oder über denselben, seine Herkunft usw. genauere Auskunft
gibt.»

Diesem Anerbieten folgte eine genaue Beschreibung Olivers und Mr.
Brownlows Adresse. Bumble las dreimal mit großem Bedacht, faßte darauf
rasch seinen Entschluß und war nach wenigen Minuten auf dem Wege nach
Pentonville. Im Hause Mr. Brownlows angelangt, kündigte er sogleich
den Zweck seines Besuchs an. Frau Bedwin war außer sich vor Freude und
Rührung, erklärte, es immer gewußt und gesagt zu haben, daß Oliver bald
wiedergefunden werden würde, brach in Tränen aus, und die Magd eilte
zu Mr. Brownlow hinauf, der ihr gebot, den Angemeldeten augenblicklich
hereinzuführen.

Bumble trat ein, und Mr. Grimwig, der sich zufällig bei seinem
Freunde befand, faßte ihn scharf in das Auge und rief aus: «Ein
Kirchspieldiener -- so wahr ich lebe, ein Kirchspieldiener!»

«Ich bitte, lieber Freund, jetzt keine Unterbrechung», sagte Brownlow.
«Setzen Sie sich, Sir. -- Sie kommen zu mir infolge der Anzeige, die
ich in verschiedenen Blättern habe einrücken lassen?»

«Ja, Sir.»

«Sie sind ein Kirchspieldiener?»

«Ja, Sir», erwiderte Bumble stolz.

«Wissen Sie, wo sich das arme Kind gegenwärtig befindet?» fragte
Brownlow ziemlich ungeduldig.

«Nein, Sir.»

«Was wissen Sie denn aber von ihm? Reden Sie, wenn Sie etwas zu sagen
haben. Was wissen Sie von ihm?»

«Sie werden wohl eben nicht viel Gutes von ihm wissen!» fiel Grimwig
kaustisch ein, nachdem er Bumbles Mienen sorgfältig geprüft hatte.

Bumble erkannte sogleich mit großem Scharfsinn den Wunsch des Herrn,
Ungünstiges über Oliver zu vernehmen, und antwortete durch ein
feierlich-bedenkliches Kopfschütteln.

«Sehen Sie wohl?» sagte Grimwig zu Brownlow mit einem triumphierenden
Blicke.

Brownlow sah Bumble besorgt an, und forderte ihn auf, was er von
Oliver wüßte, in möglichst kurzen Worten mitzuteilen. Bumble räusperte
sich und begann. Er sprach mit umständlicher Weitschweifigkeit; der
kurze Sinn von allem, was er vorbrachte, war, Oliver sei ein armer
Kirchspielknabe von armen und lasterhaften Eltern, habe von seiner
Geburt an nur Falschheit, Bosheit und Undankbarkeit gezeigt und seiner
Gottlosigkeit dadurch die Krone aufgesetzt, daß er einen mörderischen
und feigherzigen Angriff auf einen harmlosen Knaben gemacht habe und
darauf seinem Lehrherrn entlaufen sei.

«Ich fürchte, daß Ihre Angaben nur zu wahr sind», sagte Brownlow
traurig; «hier sind die fünf Guineen. Ich würde Ihnen gern dreimal so
viel gegeben haben, wenn Sie mir etwas Vorteilhafteres über den Knaben
hätten sagen können.»

Hätte Brownlow das früher gesagt, so würde Bumble seinem Bericht
wahrscheinlich eine ganz andere Färbung gegeben haben. Es war jedoch zu
spät, er schüttelte daher mit bedeutsamer Miene den Kopf, steckte die
fünf Guineen ein und ging.

Mr. Brownlow war so niedergeschlagen, daß selbst Grimwig ihn nicht
noch mehr betrüben mochte. Er zog endlich heftig die Klingelschnur.
«Frau Bedwin,» sagte er, als die Haushälterin eintrat, «der Knabe, der
Oliver, war ein Betrüger.»

«Das kann nicht sein, Sir; kann nicht sein», entgegnete Frau Bedwin
nachdrücklich.

«Ich sage Ihnen aber, daß es so ist. Wir haben soeben einen genauen
Bericht über ihn angehört. Er ist von seiner ersten Kindheit an durch
und durch verderbt gewesen.»

«Und ich glaube es doch nicht, Sir -- nimmermehr, Sir», erwiderte Frau
Bedwin bestimmt.

«Ihr alten Weiber glaubt an nichts als an Quacksalber und
Lügengeschichten», fiel Grimwig mürrisch ein. «Ich hab's von Anfang an
gewußt. Warum hörten Sie nicht sogleich auf meine Meinung und meinen
Rat? Sie würden es sicher getan haben, wenn der kleine Schelm nicht am
Fieber krank gelegen hätte!»

«Er war kein Schelm,» entgegnete Frau Bedwin sehr unwillig, «sondern
ein sehr liebes, gutes Kind. Ich verstehe mich auf Kinder sehr wohl,
Sir, seit vierzehn Jahren, Sir; und wer nie Kinder gehabt hat, darf gar
nicht mitreden über sie -- das ist meine Meinung, Sir!»

Mr. Grimwig lächelte nur, und Frau Bedwin war im Begriff, fortzufahren,
allein Brownlow kam ihr zuvor.

«Schweigen Sie!» sagte er mit einer Entrüstung in Ton und Mienen, die
freilich seinen Gefühlen vollkommen fremd war. «Sie erwähnen den Knaben
nie wieder; ich habe geklingelt, um Ihnen das zu sagen. Hören Sie --
nie -- niemals, und unter keinerlei Vorwande. Sie können gehen -- und
wohl zu merken, ich habe im Ernst gesprochen!»

In Mr. Brownlows Hause waren betrübte Herzen an diesem Abende, und
Oliver zagte das Herz gleichfalls, als er seiner gütigen Beschützer und
Freunde gedachte. Es war indes gut für ihn, daß er nicht wußte, was sie
über ihn gehört; er hätte die Nacht vielleicht nicht überlebt.




18. Kapitel.

    Wie Oliver seine Zeit in der sittenverbessernden Gesellschaft
    seiner achtungswürdigen Freunde zubrachte.


Als am folgenden Morgen der Baldowerer und Charley Bates zu ihren
gewöhnlichen Geschäften ausgegangen waren, benutzte Mr. Fagin die
Gelegenheit, Oliver einen langen Sermon über die schreiende Sünde
der Undankbarkeit zu halten, deren er sich, wie ihm Fagin klärlich
bewies, in einem sehr hohen Maße schuldig gemacht, indem er sich
absichtlich von seinen liebevollen und treuen Freunden entfernt, ja
sogar ihnen zu entfliehen versucht habe, nachdem sie so viele Mühe
und Kosten aufgewandt hätten, ihn wieder zurückzubringen. Der alte
Herr legte großes Gewicht auf den Umstand, daß er Oliver zu sich
genommen und verpflegt habe, als derselbe in Gefahr gewesen wäre,
Hungers zu sterben, und erzählte ihm die ergreifende und schreckliche
Geschichte eines jungen Burschen, dem er unter ähnlichen Umständen
aus gewohnter Menschenfreundlichkeit seinen Beistand habe angedeihen
lassen, der sich aber des ihm erwiesenen Vertrauens unwürdig gezeigt,
sich mit der Polizei in Rapport zu setzen versucht habe und im
Old-Bailey-Gerichtshofe verurteilt und gehenkt worden sei. Der alte
Herr bemühte sich durchaus nicht, seinen Anteil an der Katastrophe
zu verheimlichen, sondern beklagte es mit Tränen in den Augen,
daß es durch die Verkehrtheit und Verräterei des jungen Burschen
nötig geworden, ihn als ein Opfer fallen zu lassen, und demnach mit
Zeugnissen gegen ihn aufzutreten, die, wenn auch nicht vollkommen
in der Wahrheit begründet, doch unumgänglich gewesen wären, wenn
seine (Fagins) und einiger erlesener Freunde Sicherheit nicht hätte
gefährdet werden sollen. Der alte Herr schloß damit, daß er ein sehr
unerfreuliches Gemälde von den Unannehmlichkeiten des Gehenktwerdens
entwarf und mit großer Freundschaftlichkeit und Höflichkeit die
Hoffnung ausdrückte, niemals genötigt zu werden, Oliver Twist einer so
widerwärtigen Operation zu unterwerfen.

Dem kleinen Oliver erstarrte das Blut in den Adern, während er den
Worten des Juden zuhörte. Die darin enthaltenen dunklen Drohungen waren
ihm nicht ganz unverständlich. Er wußte bereits, daß die Gerechtigkeit
selbst den Unschuldigen für schuldig halten konnte, wenn er sich mit
dem Schuldigen in Gemeinschaft befunden hatte; und daß tief angelegte
Pläne, unbequeme Mitwisser oder zum Schwatzen Geneigte zu verderben,
von dem alten Juden wirklich geschmiedet und ausgeführt wären, dünkte
ihn keineswegs unwahrscheinlich, als er sich des Streites entsann, den
Fagin mit Sikes gehabt. Als er furchtsam die Augen aufschlug und seine
Blicke denen des Juden begegneten, fühlte er, daß seine Blässe und
sein Zittern dem schlauen Bösewicht nicht entgangen waren und daß sich
derselbe innerlich darüber freute.

Der Jude lächelte boshaft, klopfte Oliver die Wangen und sagte ihm,
wenn er sich ruhig verhielte und sich des Geschäfts annähme, so würden
sie sicher noch sehr gute Freunde werden. Er griff darauf zum Hute, zog
einen alten, geflickten Oberrock an, ging hinaus und verschloß die Tür
hinter sich.

So blieb sich Oliver während des ganzen Tages und während noch vieler
nachfolgender Tage vom frühen Morgen bis Mitternacht selbst überlassen,
und die langen Stunden vergingen ihm gar traurig, denn er gedachte
natürlich fortwährend seiner gütigen Freunde in Pentonville und der
Meinung, welche sie von ihm gefaßt haben müßten. Am siebenten oder
achten Tage ließ der Jude die Tür des Zimmers unverschlossen, und
Oliver durfte frei im Hause umhergehen. -- Das ganze Haus war äußerst
schmutzig und öde; die Zimmer im oberen Stockwerke waren ohne Mobilien,
geschwärzt und mit Spinngeweben überdeckt; indes schloß Oliver aus dem
Täfelwerke und den Resten alter Tapeten und anderer Verzierungen, daß
sie vor langer Zeit von reichen Leuten bewohnt gewesen sein müßten,
so kläglich sie auch jetzt aussahen. Oft, wenn er leise in ein Zimmer
eintrat, liefen die Mäuse erschreckt in ihre Löcher zurück; sonst aber
sah und hörte er kein lebendiges Wesen, und manches Mal, wenn er es
müde war, aus einem Gemach in das andere zu wandern, schmiegte er sich
in den Winkel des Flurs an der Haustür, um den Menschen so nahe wie
möglich zu sein, und erwartete horchend und mit Beben die Rückkehr des
Juden oder der Knaben.

In allen Zimmern waren die Fensterläden fest mit Schrauben verwahrt und
ließen nur wenig Licht durch kleine, runde Löcher ein, was die Zimmer
noch düsterer machte und sie mit seltsamen Schattengestalten füllte.
Ein hinteres Dachstübchen hatte ein mit starken Stäben verwahrtes
Fenster ohne Läden. Oliver schaute stundenlang traurig hinaus, obwohl
er nichts sehen konnte als eine verworrene, gedrängte Masse von
Dächern, geschwärzten Schornsteinen und Giebeln. Bisweilen zeigte sich
auf ein paar Augenblicke in der Dachluke eines fernen Hauses ein nur
undeutlich zu erkennendes Gesicht; allein es verschwand bald, und da
das Fenster von Olivers Observatorium vernagelt und durch Regen und
Rauch von Jahren trüb und blind gemacht worden war, so war es ihm nur
möglich, die Formen ferner Gegenstände undeutlich zu erkennen, und er
konnte nicht daran denken, sich bemerkbar zu machen, zumal auch die
Nachbarschaft sicher nicht die achtbarste und vertrauenerweckendste war.

Eines Nachmittags kehrten der Baldowerer und Charley Bates nach Hause
zurück, um sich auf eine Abendunternehmung vorzubereiten, die es
erfordern mochte, daß sie sich sorgfältiger als gewöhnlich ankleideten.
Der Baldowerer gebot Oliver, ihm die Stiefel zu reinigen, und Oliver
war froh, nur einmal Menschen zu sehen und sich nützlich machen zu
können, wenn es ohne Verletzung der Redlichkeit geschehen konnte.
Während er beschäftigt war, dem Geheiß Folge zu leisten, wobei Jack
auf einem Tische saß, blickte der junge Gentleman zu ihm hernieder,
seufzte und sagte halb zerstreut und halb zu Charley Bates: «'s ist
doch Jammer und Schade, daß er kein Kochemer ist.»

«Ah,» sagte Charley Bates, «er weiß nicht, was ihm gut ist.»

«Du weißt wohl nicht mal, Oliver, was ein Kochemer ist?» fragte der
Baldowerer.

«Ich glaube es zu wissen», erwiderte Oliver schüchtern; «ein Dieb --
bist du nicht ein Dieb?»

«Ja,» sagte Jack, «und ich rechn' es mir zur Ehre. Ich bin ä Dieb;
Charley ist's, Fagin ist's, Sikes ist's; Nancy und Betsy sind
gleichfalls Diebinnen. Wir sind samt und sonders Diebe, bis herunter zu
Sikes Hund, und der geht noch über uns alle.»

«Und ist kein Angeber», bemerkte Charley Bates.

«Er würde in der Zeugenloge nicht mal bellen, um sich nicht zu verraten
oder verdächtig zu machen», fuhr Jack fort. «Doch das hat nichts zu
schaffen mit unserm Musjö Grün.»

«Warum begibst du dich nicht unter Fagins Oberbefehl, Oliver?» fiel
Charley ein.

«Könntest doch dein Glück so schön machen», setzte Jack hinzu, «und
von deinem Gelde leben wie ein Gentleman, wie ich's zu tun denke im
nächstkommenden fünften Schaltjahr und am zweiundvierzigsten Dienstag
in der Fastenwoche.»

«Es gefällt mir nicht», sagte Oliver furchtsam. «Ich wollte, daß Fagin
mich fortgehen ließe.»

«Das wird Fagin bleiben lassen», bemerkte Charley.

Oliver wußte dies nur zu gut, und in der Meinung, es möchte gefährlich
sein, seine Gedanken noch offener auszusprechen, fuhr er seufzend in
seinem Geschäfte fort.

«Schäme dich!» hub der Baldowerer wieder an. «Hast du denn gar kein
Ehrgefühl? Ich möchte um nichts in der Welt meinen Freunden zur Last
fallen, am wenigsten, ohne 'nen Finger zu rühren, um ihnen zum
wenigsten meine Erkenntlichkeit zu beweisen.»

«Es ist wahrhaftig zu gemein und niedrig», sagte Charley Bates, einige
seidene Taschentücher hervorziehend und in eine Kommode legend.

«Es wäre mir ganz unmöglich!» rief Jack Dawkins, sich in die Brust
werfend, aus.

«Und doch könnt ihr eure Freunde im Stich lassen», bemerkte Oliver mit
einem halben Lächeln, «und zusehen, daß sie für Dinge bestraft werden,
die ihr getan habt.»

«Es geschah bloß aus Rücksicht gegen Fagin», erwiderte Jack kaltblütig.
«Die Schoderer[R] wissen, daß wir gemeinschaftlich arbeiten, und er
hätte in Ungelegenheit kommen können, wenn wir nicht davongelaufen
wären. Schau hier», setzte er hinzu, griff in die Tasche und zeigte
Oliver eine Handvoll Schillinge und Halbpence. «Wir führen ä flottes
Leben, und was tut's, woher das Geld dazu kommt? Da, nimm hin; wo's
her ist, da ist noch mehr von der Sorte. Du willst nicht? O Dümmling,
Dümmling aller Dümmlinge!»

  [R] Gerichtsdiener.

«Er ist ä Bösewicht, nicht wahr, Oliver?» fiel Charley Bates ein. «Er
wird noch geschnürt werden, nicht wahr?»

«Ich weiß nicht, was das ist», sagte Oliver.

Charley Bates nahm sein Taschentuch, knüpfte es sich um den Hals und
stellte die Hängeoperation pantomimisch und vollkommen kunstgerecht
dar. «Das ist's», sagte er endlich unter schallendem Gelächter.

«Du bist schlecht erzogen,» bemerkte Jack Dawkins ernsthaft; «indes
wird Fagin doch schon noch etwas aus dir machen, oder du wärst der
erste, der sich ganz unbrauchbar gezeigt. Also fang nur, je eher, desto
lieber, an, denn du wirst mitarbeiten im Geschäft, eh' du's meinst, und
verlierst nur Zeit, Oliver.»

Charley Bates fügte noch mehrere moralische Betrachtungen hinzu,
schilderte mit glühenden Farben die zahllosen Annehmlichkeiten des
Lebens, das sie, er und Jack, führten, und bemühte sich mit einem Worte
auf das eifrigste, Oliver zu überzeugen, daß er nichts Besseres tun
könne, als baldmöglichst um Fagins Gunst durch dieselben Mittel zu
werben, die er und Jack zum gleichen Zwecke angewendet.

«Und vor allen Dingen, Nolly[S],» sagte Jack, als sie den Juden kommen
hörten, «bedenk' das: nimmst du keine Schneichen und Zwiebeln --»

  [S] Oliver.

«Was hilft's, daß du so zu ihm redest?» unterbrach Charley; «weiß er
doch nicht, was du damit sagen willst!»

«Nimmst du keine Taschentücher und Uhren,» fuhr der Baldowerer, zu
Olivers Fassungskraft sich herablassend, fort, «so tut's der erste
beste andere, und der hat was davon, und du hast nischt, da du doch ein
ebenso gutes Recht dazu hast.»

«'s ist ganz klar -- ja, ja -- ganz klar,» sagte der Jude, der
unbemerkt von Oliver eingetreten war, «klar wie die Sonne, mein Kind.
Glaub' dem Baldowerer; er kennt den Katechismus seines Geschäfts aufs
Haar.»

Das Gespräch wurde indes für jetzt abgebrochen, da Fagin mit Miß Betsy
und einem Gentleman angelangt war, den Oliver noch nicht gesehen hatte
und den der Baldowerer Tom Chitling nannte, als er eintrat, nachdem
er draußen ein wenig verweilt, um mit der Dame einige Galanterien zu
wechseln.

Tom Chitling war älter an Jahren als der Baldowerer, da er etwa
achtzehn Winter zählen mochte, bezeigte demselben aber eine
Ehrerbietung, woraus man klärlich sah, daß er sich bewußt war, an
Genie und Geschäftserfahrung ihm untergeordnet zu sein. Tom hatte
kleine, blinzelnde Augen und ein pockennarbiges Gesicht und trug eine
Pelzkappe, eine Jacke aus dunklem Tuch, fettige Barchenthosen und eine
Schürze. Er sah in der Tat ziemlich abgerissen aus, entschuldigte sich
jedoch bei der Gesellschaft damit, daß «seine Zeit» erst seit einer
Stunde aus gewesen sei, daß er seine Uniform sechs Wochen getragen
und noch nicht daran habe denken können, die Garderobe zu wechseln.
Er schloß mit der Bemerkung, daß er zweiundvierzig Tage angestrengt
gearbeitet, und «bersten wolle, wenn er in der ganzen Zeit 'nen Tropfen
gekostet und nicht so trocken sei wie ein Sandfaß».

«Was meinst du, Oliver, woher der junge Mensch wohl kommt?» fragte der
Jude grinsend, während Charley eine Branntweinflasche auf den Tisch
stellte.

«Ich -- ich kann's nicht sagen, Sir», erwiderte Oliver.

«Wer ist denn der?» fragte Tom Chitling, Oliver verächtlich anblickend.

«Ein junger Freund von mir, mein Lieber», antwortete Fagin.

«Dann hat er's gut genug», bemerkte Tom, dem Juden einen bedeutsamen
Blick zuwerfend. «Kümmere dich nicht darum, Bursch, woher ich komme; es
gilt 'ne Krone, wirst bald genug selber da sein!»

Es wurde gelacht, Fagin flüsterte mit Tom, alle versammelten sich
am Kamine, der Jude forderte Oliver auf, sich zu ihm zu setzen, und
lenkte das Gespräch auf Gegenstände, von welchen er erwarten konnte,
daß seine Zuhörer den lebhaftesten Anteil daran nehmen würden; nämlich
die großen Vorteile des Geschäfts, die Talente des Baldowerers, die
Liebenswürdigkeit Charleys und die Freigebigkeit Fagins. Als sie
erschöpft waren und Tom Chitling gleichfalls Zeichen der Erschöpftheit
an den Tag legte (denn das Besserungshaus ermüdet sehr nach einigen
Wochen), entfernte sich Miß Betsy, und die übrigen begaben sich zur
Ruhe.

Von diesem Tage an wurde Oliver nur noch selten allein gelassen und
in eine fortwährende enge Verbindung mit Jack und Charley gebracht,
die mit dem Juden täglich das alte Spiel spielten -- Fagin wußte am
besten, ob zu ihrer eigenen oder Olivers Belehrung und Vervollkommnung.
Zu anderen Zeiten erzählte ihnen Fagin Geschichten von Diebstählen und
Räubereien, die er in seinen jüngeren Tagen begangen, und mischte so
viel Merkwürdiges, Spaßhaftes und Drolliges ein, daß Oliver sich oft
nicht enthalten konnte, herzlich zu lachen und den Beweis zu liefern,
daß er trotz seiner besseren Gefühle Wohlgefallen an diesen Geschichten
fand.

Kurzum, der schlaue alte Jude hatte den Knaben sozusagen im Netze und
war, nachdem er ihn durch Einsamkeit und die Qual derselben dahin
gebracht, jede Gesellschaft seinen traurigen Gedanken in einem so öden,
finsteren Hause vorzuziehen, eifrig darüber aus, seinem Herzen das Gift
langsam einzuflößen, das, wie er hoffte, die Farbe desselben verändern
und es für immer schwärzen sollte.




19. Kapitel.

    In welchem ein verhängnisvoller Plan besprochen und beschlossen
    wird.


Es war ein kalter, feuchter und stürmischer Abend, als der Jude seinen
eingeschrumpften Leib in einen Oberrock einhüllte, den Kragen über die
Ohren zog, so daß von seinem Gesicht nur die Augen zu sehen waren, und
sich aus seiner Höhle entfernte. Er blieb vor der Haustür stehen, bis
sie inwendig verschlossen und verriegelt war, und eilte darauf mit
leisen und flüchtigen Schritten die Straße hinunter.

Das Haus, in welches Oliver gebracht worden war, befand sich nahe bei
Whitechapel; der Jude stand an der nächsten Ecke ein paar Augenblicke
still, schaute forschend umher und schlug sodann die Richtung nach
Spitalfields ein.

Auf dem Pflaster lag dicker Schlamm, und ein dichter Nebel machte die
Dunkelheit noch dunkler. Für den Ausflug eines dämonischen Wesens,
wie es der Jude war, konnten Zeit, Wetter und alle Umgebungen nicht
passender sein. Der greuliche Alte glich, während er verstohlen durch
Nacht und Nebel und Kot dahineilte, einem ekelhaften Gewürm, das in
nächtlicher Finsternis aus seinem Verstecke herauskriecht, um wühlend
im Schlamme ein leckeres Mahl nach seiner Art zu halten.

Er setzte seinen Weg durch viele enge und winklige Gassen fort, bis er
Bethnal Green erreichte, wandte sich dann nach links und verschwand in
einem wahrhaften Labyrinth schmutziger Winkel, Straßen und Gassen jenes
zahlreich bevölkerten Stadtviertels, ohne jedoch ein einziges Mal zu
irren oder fehlzugehen, lenkte endlich in eine Sackgasse ein, klopfte
an die Tür eines Hauses und wurde, nachdem er ein paar Worte durch das
Schlüsselloch geflüstert, eingelassen und hinaufgeführt.

Als er auf den Griff einer Tür faßte, knurrte ein Hund, und eine grobe
Mannsstimme fragte, wer da wäre.

«Ich bin's, Bill, ich, mein Lieber», antwortete der Jude hineinschauend.

«So bringt Eur'n Leichnam 'rein», sagte Sikes. «Lieg' still, dumme
Bestie! Kennst den Teufel nicht, wenn er'n Überrock anhat?»

Der Hund schien in der Tat durch Fagins Verhüllung getäuscht zu sein;
denn sobald der Jude den Oberrock aufknöpfte, legte er sich, mit dem
Schweife wedelnd, wieder nieder.

«Nun?» sagte Sikes.

«Ja -- nun», erwiderte der Jude. «Ah, Nancy.»

Er schien etwas verlegen und zweifelhaft zu sein, wie er von Miß Nancy
empfangen werden würde, denn er hatte seine junge Freundin seit dem
Abend noch nicht wiedergesehen, an welchem sie so leidenschaftlich für
Oliver aufgetreten war. Das Benehmen der jungen Dame machte jedoch
bald aller Ungewißheit ein Ende. Sie schob ihren Stuhl zur Seite und
forderte Fagin auf, ohne Groll oder noch viel Worte sich mit an den
Kamin zu setzen, denn es wäre ein kalter Abend.

«Ja, 's ist bitter kalt, liebe Nancy», sagte Fagin und begann seine
knöchernen Hände über dem Feuer zu wärmen. «'s ist, als wenn der Wind
einem wehte durch und durch bis ins Innerste.»

«Das muß wirklich scharf sein, was bis an dein Herz dringt», bemerkte
Sikes. «Gib ihm 'nen Tropfen zu trinken, Nancy. Alle Donnerwetter, mach
geschwind! Man wird ganz übel davon, das alte Gerippe so schaudern zu
sehn wie'n häßliches Gespenst, das eben aus'm Grabe gestiegen ist.»

Nancy holte schnell eine Flasche aus dem Schranke; Sikes schenkte ein
Glas Branntwein ein und hieß den Juden es austrinken; Fagin berührte es
jedoch nur mit den Lippen und setzte es wieder auf den Tisch.

«Ausgetrunken, Spitzbube!» rief Sikes.

«Habe schon genug, danke, Bill!»

«Wie -- was? Fürchtest dich, daß wir dir ä Streich spielen?» fragte
Sikes, seine Augen scharf auf den Juden richtend.

Mit einem heiseren, verächtlichen Brummen ergriff Mr. Sikes das Glas
und goß den Inhalt in die Asche; dann füllte er es von neuem und
stürzte es hinunter.

Fagin blickte im Zimmer umher, nicht aus Neugierde, denn es war ihm
wohlbekannt, sondern unruhig, verstohlen, argwöhnisch, wie es ihm zur
Gewohnheit geworden war. Das Gemach war sehr schlecht möbliert. Nur der
Inhalt des Schrankes schien anzudeuten, daß es von einem gewöhnlichen
Arbeiter bewohnt würde; auch sah man nichts Verdächtiges, mit Ausnahme
einiger schwerer Knüttel, die in einem Winkel standen, und eines
«Lebensretters», der über dem Kaminsimse hing.

«Was hast du zu sagen, verdammter Jude?» fragte Sikes. «Weshalb bist du
hergeschlichen?»

«Wegen des Bayes[T] in Chertsey, Bill», erwiderte der Jude, dicht zu
ihm rückend und flüsternd.

  [T] Haus.

«Nun -- und was weiter?»

«Ah -- Ihr wißt ja recht gut, was ich meine, Bill. Nicht wahr, Nancy,
er weiß es recht gut?»

«Nein, er weiß es nicht», fiel Sikes höhnisch ein, «oder will es nicht
wissen, was dasselbe ist. Sprich rein 'raus, nenn' die Dinge beim
rechten Namen und stell' dich nicht an, als wenn du nicht der erste
gewesen wärst, der an den Einbruch gedacht hat.»

«Pst, Bill, pst!» sagte Fagin, der sich vergebens bemüht hatte, Sikes
zum Stillschweigen zu bringen; «es wird uns jemand hören, mein Lieber,
es wird uns jemand hören!»

«Laß hören, wer will!» tobte Sikes; «'s ist mir alles gleich.»

Er sprach jedoch die letzten Worte schon weniger laut und heftig, da
ihm der Gedanke kam, daß es doch *nicht* gleich wäre oder sein könnte.

«Seid doch ruhig, Bill,» sagte der Jude besänftigend. «Es war ja nur
meine Vorsicht -- weiter nichts. Also wegen des Bayes in Chertsey, mein
Lieber. Wann soll's sein, Bill -- wann soll's sein? Solch Silberzeug,
Bill, solch Silberzeug!» setzte er händereibend und mit leuchtenden
Augen hinzu.

«Gar nicht», erwiderte Sikes trocken.

«Gar nicht?» wiederholte der Jude und lehnte sich erstaunt auf seinem
Stuhle zurück.

«Nein, gar nicht», sagte Sikes; «zum wenigsten kann's nicht so
ausgeführt werden, wie wir meinten.»

«Dann ist's nicht geschickt und ordentlich angegriffen», versetzte der
Jude, vor Verdruß erblassend. «Aber Ihr spaßt nur, Bill.»

«Ich lasse mich lieber hängen, als daß ich mit dir spaße, altes
Gerippe. Toby Crackit hat sich seit vierzehn Tagen die erdenklichste
Mühe gegeben, aber keinen von der Dienerschaft --»

«Ihr wollt doch nicht sagen, Bill,» unterbrach ihn der Jude ungeduldig,
doch aber ruhiger in dem Maße, als Sikes wieder heftig zu werden
anfing; «Ihr wollt doch nicht sagen, daß keiner von den beiden
Bedienten könnte werden gewonnen, zu machen Kippe?»

«Das will ich allerdings sagen», antwortete Sikes. «Sie sind seit
zwanzig Jahren bei der alten Frau im Dienst gewesen und würden's nicht
tun für fünfhundert Pfund.»

«Aber die weibliche Dienerschaft, mein Lieber -- läßt sich die auch
nicht beschwatzen?»

«Nein!»

«Wie -- auch nicht vom schmucken, geriebenen Toby Crackit?» entgegnete
der Jude ungläubig. «Bedenkt doch nur, wie die Weibsen sind, Bill!»

«Nein, auch nicht von Toby Crackit», erwiderte Sikes. «Er hat die ganze
Zeit, daß er's Bayes umschlichen, falsche Knebelbärte und 'ne gelbe
Weste getragen; hat aber alles nicht helfen wollen.»

«Er hätt's versuchen sollen mit 'nem Schnurrbart und Soldatenhosen,
mein Lieber», sagte der Jude nach einigem Besinnen.

«Das hat er auch schon getan, und 's ist ebenso vergeblich gewesen.»

Der Jude machte eine verdrießliche und verlegene Miene dazu, versank
auf ein paar Minuten in tiefes Nachsinnen und sagte endlich mit einem
schweren Seufzer, wenn man sich auf Toby Crackits Berichte verlassen
könnte, so fürchte er, daß der Plan aufgegeben werden müsse. «'s ist
aber sehr betrübend, Bill,» setzte er, die Hände auf die Knie stützend,
hinzu, «so viel zu verlieren, wenn man einmal den Sinn hat gesetzt
darauf.»

«Freilich,» sagte Sikes, «'s ist ganz verdammt ärgerlich!»

Es folgte ein langes Stillschweigen. Der Jude war in tiefe Gedanken
verloren, und sein Gesicht nahm einen Ausdruck wahrhaft satanischer
Spitzbüberei an. Sikes blickte ihn von Zeit zu Zeit verstohlen von
der Seite an, und Nancy heftete, aus Furcht, den Wohnungsinhaber
zu erzürnen, die Augen auf das Feuer, als wenn sie bei allem, was
gesprochen worden, taub gewesen wäre.

«Fagin,» unterbrach Sikes endlich die allgemeine Stille, «schafft's
fünfzig Füchse extra, wenn's durch Einbruch vollbracht wird?»

«Ja!» rief der Jude, wie aus einem Traume erwachend.

«Abgemacht?» fragte Sikes.

«Ja, mein Lieber,» erwiderte der Jude, indem er ihm die Hand reichte;
und jede Muskel seines Gesichts gab Zeugnis, wie freudig und lebhaft
er durch diese Frage überrascht worden war.

Sikes schob die Hand des Juden verächtlich zurück und fuhr fort: «Dann
mag's geschehen, sobald du willst, Alter. Toby und ich sind vorgestern
über die Gartenmauer g'wesen und haben die Türen und Fensterläden
untersucht. Die Bayes ist nachts verrammelt wie'n Dobes; wir haben
aber 'ne Stelle gefunden, wo wir leise und mit Sicherheit schränken[U]
können.»

  [U] einbrechen.

«Wo ist denn die Stelle, Sikes?» fragte der Jude sehr gespannt.

«Man geht über den Rasenplatz,» flüsterte Sikes, «und dann --»

«Nun, und dann?» unterbrach ihn der Jude, sich ungeduldig vorbeugend.

«Dann --» sagte der Schränker, brach jedoch kurz ab, denn Nancy gab
ihm, kaum den Kopf bewegend, einen Wink, nach des Juden Gesicht zu
sehen. «'s ist ganz gleich, wo die Stelle ist», fuhr er fort. «Ich
weiß, daß du's nicht kannst ohne mich; aber man tut wohl daran, sich
auf Nummer Sicher zu setzen, wenn man mit dir zu tun hat.»

«Nach Eurem Belieben, Bill, nach Eurem Belieben», erwiderte der Jude,
sich auf die Lippen beißend. «Könnt Ihr's mit Toby allein, und braucht
Ihr weiter keinen Beistand?»

«Nein; bloß ein Dreheisen und 'nen Knaben. Das Eisen haben wir, den
Buben mußt du uns schaffen.»

«'nen Knaben!» rief der Jude aus. «Ah, dann ist's ein Paneel -- wie?»

«Es kann dir gleichviel sein, was es ist», erwiderte Sikes. «Ich
brauche 'nen Buben, und er darf nicht groß sein. Wenn mir nur nicht der
von Ned, dem Schornsteinfeger, durch die Lappen 'gangen wäre! Er hielt
ihn mit Absicht klein und schmächtig und lieh ihn aus für'n Billiges.
Aber so geht's, der Vater wird gerumpelt[V], und wie der Blitz ist
der Verein für verlassene Kinder da und nimmt den Jungen aus 'nem
Geschäft, darin er Geld hätte verdienen können, lehrt ihn Lesen und
Schreiben, und der Bube wird dann Lehrling, Gesell, endlich Meister,»
sagte Sikes mit steigendem Zorn über einen so unrechtmäßigen Verlauf,
«und so geht's mit den meisten; und hätten sie immer Geld genug, was
sie Gott Lob und Dank nicht haben, so würden wir nach ein paar Jahren
keinen einzigen Jungen mehr im Geschäft halten.»

  [V] deportiert.

«Ja, ja», stimmte der Jude ein, der unterdes überlegt und nur die
letzten Worte gehört hatte. «Bill!»

«Was gibt's?»

Der Jude deutete verstohlen auf Nancy hin, die noch immer in das Feuer
schaute, und gab Sikes durch Zeichen seinen Wunsch zu erkennen, mit ihm
allein gelassen zu werden. Sikes zuckte ungeduldig die Achseln, als wenn
er die Vorsicht für überflüssig hielte, forderte indes Nancy auf, ihm
einen Krug Bier zu holen.

«Ihr seid nicht durstig, Bill», sagte Nancy mit der vollkommensten Ruhe
und schlug die Arme übereinander.

«Ich sage dir, ich bin durstig!» entgegnete Sikes.

«Dummes Zeug! Fahrt fort, Fagin. Ich weiß, was er sagen will, Bill; ich
kann's auch hören.»

Der Jude zögerte, und Sikes sah etwas verwundert bald ihn, bald das
Mädchen an.

«Brauchst dich vor dem alten Mädchen nicht zu scheuen, Fagin», sagte er
endlich. «Hast sie lange genug gekannt und kannst ihr trauen, oder der
Teufel müßte drin sitzen. Sie wird nicht mosern[W]; nicht wahr, Nancy?»

  [W] verraten.

«Ihr sollt's wohl meinen», erwiderte sie, ihren Stuhl an den Tisch
schiebend und den Kopf auf die Ellbogen stützend.

«Nein, nein, liebes Kind», fiel der Jude ein; «ich weiß das sehr wohl;
nur --» Er hielt wieder inne.

«Nun, was denn nur?» fragte Sikes.

«Ich weiß nur nicht, ob sie nicht vielleicht wieder werden würde
unwirsch, mein Lieber, wie vor einigen Abenden», erwiderte Fagin.

Bei diesem Geständnisse brach Nancy in ein lautes Gelächter aus,
stürzte ein Glas Branntwein hinunter, erklärte unter mehrfachen
kräftigen Beteuerungen, daß sie alles hören könne, wolle und werde und
so standhaft, mutvoll und treu sei wie eine oder einer. -- «Fagin,»
sagte sie lachend, «sprecht nur ohne Umschweife zu Bill von Oliver!»

«Ah! Du bist ein so gewitztes Mädchen, wie ich je eins gesehen»,
versicherte der Jude und klopfte sie auf die Wange. «Ja, ich wollte
wirklich sprechen von Oliver; ha, ha, ha!»

«Was ist mit ihm los?» fragte Sikes.

«Daß er der Knabe ist, den Ihr braucht, mein Lieber», erwiderte der
Jude in einem heiseren Flüstern, den Finger an die Nase legend und mit
einem fürchterlichen Grinsen.

«Der Oliver?!» rief Bill aus.

«Nimm ihn, Bill», sagte Nancy. «Ich tät's, wenn ich an deiner Stelle
wäre. Mag sein, daß er nicht so gepfifft und dreist ist wie einer
von den andern; aber das ist auch nicht nötig, wenn du ihn bloß dazu
brauchen willst, daß er dir 'ne Tür aufmacht. Verlaß dich darauf, er
ist petacht[X], Bill.»

  [X] zuverlässig.

«Ich weiß, daß er's ist», fiel Fagin ein. «Er ist in den letzten
Wochen geschult gut, und 's ist Zeit, daß er anfängt, für sein Brot zu
arbeiten; außerdem sind die andern alle zu groß.»

«Ja, die rechte Größe hat er», bemerkte Sikes nachdenklich.

«Und er wird alles tun, wozu Ihr ihn nötig habt, Bill», sagte der Jude.
«Er kann nicht anders -- nämlich, wenn Ihr ihn nur genug haltet in
Furcht und Schrecken.»

«Das könnte geschehen -- und nicht bloß zum Spaß. Ist was nicht richtig
mit ihm, wenn wir einmal erst am Werk sind -- alle Teufel! -- so siehst
du ihn nicht lebendig wieder, Fagin. Bedenk' das, eh' du ihn schickst.»

Er hatte ein schweres Brecheisen unter dem Bette hervorgezogen und
schüttelte es unter drohenden Gebärden.

«Ich habe alles bedacht», erwiderte der Jude entschlossen. «Ich hab'
ihn beobachtet, meine Lieben, wie ein Falke die Augen auf ihn gehabt.
Laßt ihn nur erst wissen, daß er einer der Unsrigen ist; laßt ihn nur
erst wissen, daß er gewesen ist ein Dieb, und er ist unser -- unser auf
sein Leben lang! Oho! Es hätte nicht besser können kommen!» Er kreuzte
die Arme über der Brust, zog den Kopf zwischen die Schultern und
umarmte sich gleichsam selbst vor Behagen und Freude.

«Unser!» höhnte Sikes. «Du willst sagen: dein.»

«Könnte vielleicht sein, mein Lieber», sagte der Jude kichernd. «Wenn
Ihr's so wollt, Bill, mein.»

Sikes warf seinem angenehmen Freunde finster-grollende Blicke zu. «Und
warum bemühst du dich denn so sehr um das Kreidegesicht,» sagte er,
«da du doch weißt, daß jede Nacht fünfzig Buben im Common Garden[Y]
dormen[Z], unter denen du die Wahl hast?»

  [Y] Coventgarden.

  [Z] schlafen.

«Weil ich sie nicht gebrauchen kann, mein Lieber», erwiderte der
Jude ein wenig verwirrt. «Sie sind's nicht wert, daß man's versucht
mit ihnen, denn wenn sie in Ungelegenheiten geraten, steht ihnen
geschrieben auf der Stirn, was sie sind und was sie haben getan,
und sie gehen mir alle kapores. Aber mit diesem Knaben, wenn er nur
gebraucht wird geschickt, kann ich ausrichten mehr als mit zwanzig
von den anderen. Außerdem,» fügte er wieder in vollkommener Fassung
hinzu, «außerdem *haben wir* ihn dann fest jetzt, wenn er uns wieder
entwischen könnte, und *er* muß bleiben mit uns im selben Boot,
gleichviel wie er gekommen ist hinein; ich habe Macht genug über ihn,
wenn er nur ein einziges Mal ist gewesen bei 'nem Schränken -- mehr
brauch' ich nicht. Und wieviel ist das besser, als wenn wir müßten den
armen, kleinen Knaben über die Seite schaffen, was würde gefährlich
sein -- und wodurch wir verlieren würden viel!»

Sikes schwebte eine starke Mißbilligung bei Fagins plötzlicher
Anwandlung von Menschlichkeit auf den Lippen, Nancy kam ihm jedoch
durch die Frage zuvor, wann der Einbruch geschehen sollte.

«Ja, Bill, ja -- wann soll es sein?» fragte auch der Jude.

«Ich hab's mit Toby auf übermorgen nacht verabredet,» antwortete Sikes
mürrisch, «wenn ich ihm keine anderweitige Nachricht zugehen lasse.»

«Gut», sagte der Jude; «es wird doch kein Mondschein sein?»

«Nein», erwiderte Sikes.

«Ist auch bedacht alles wegen Fortschaffens der Sechore[AA]?» fragte
Fagin.

  [AA] Gestohlenes Gut.

Sikes nickte.

«Und wegen --»

«Ja, ja, 's ist alles verwaldiwert[AB],» unterbrach ihn Sikes; «scher
dich nur nicht weiter drum. Bring' den Buben lieber morgen abend her.
Ich werde 'ne Stunde nach Tagesanbruch auf und davon sein. Und dann
halt's Maul und stelle den Schmelztiegel bereit; das ist alles, was du
zu tun hast.»

  [AB] verabredet.

Nach einigem Hin- und Herreden, woran alle drei tätigen Anteil nahmen,
wurde beschlossen, daß Nancy am folgenden Abend Oliver herbringen
solle. Fagin hielt dafür, daß er Nancy am ersten folgen würde, wenn er
etwa abgeneigt wäre. Ebenso wurde feierlich verabredet, daß der Knabe
zum Zweck der beabsichtigten Unternehmung Sikes unbedingt übergeben
werden solle, und zwar so, daß derselbe mit ihm nach Gutdünken
verfahren dürfe, ohne dem Juden für irgendeinen Unfall, der ihn treffen
könnte, oder irgendeine Züchtigung verantwortlich zu sein, die sein
Beschützer etwa für notwendig erachten möchte; auch sollte der letztere
alle seine Angaben nach seiner Rückkehr durch Toby Crackits Zeugnis
bestätigen lassen. Sikes bekräftigte vorläufig den edlen Bund und
die Aufrichtigkeit seiner Gesinnungen durch ein Glas Branntwein nach
dem andern, was die Wirkung hatte, daß er zuerst lärmte und sodann
einschlief. Der Jude hüllte sich darauf wieder in seinen Überrock,
sagte Nancy gute Nacht und faßte sie scharf ins Auge, während sie ihm
zur Erwiderung gleichfalls wohl zu schlafen wünschte und ihre Blicke
den seinigen begegneten. Sie waren vollkommen fest und ruhig. Das
Mädchen war so treu und verläßlich in der Sache, wie Toby Crackit nur
selbst sein konnte. Er warf Sikes, unbemerkt von ihr, noch einen Blick
des Hasses und der Verachtung zu und ging, durch die Zähne murmelnd:
«So sind sie alle. Das Schlimmste an den Weibsbildern ist, daß die
größte Kleinigkeit aufweckt in ihnen ein längst vergessenes Gefühl --
und das Beste, daß es nicht währt lange. Hi, hi, hi! Ich wette 'nen
Sack voll Gold auf den Mann gegen das Kind.»

Unter diesen angenehmen Gedanken ging Fagin seines Weges durch Schmutz
und Kot hin bis zu seiner düsteren Wohnstätte. Der Baldowerer war
aufgeblieben und erwartete ungeduldig die Rückkehr des Juden.

«Ist Oliver zu Bett? Ich wünsche ihn zu sprechen», war die erste Frage,
die er tat, als beide die Treppe hinunterstiegen.

«Schon seit mehreren Stunden», versetzte der Baldowerer, indem er eine
Tür aufstieß. «Hier ist er.»

Der Knabe lag fest schlafend auf einer harten Matratze auf dem
Fußboden, so bleich vor Angst, Traurigkeit und Verlassenheit in seinem
Gefängnis, daß er Ähnlichkeit mit einem Toten hatte -- nicht mit einem
Toten, wie er im Sarge und auf der Bahre liegt, sondern mit einem,
aus dem das Leben soeben entwichen ist, wenn ein junger, edler Geist
zum Himmel entflohen ist und die schwere Luft der Welt noch keine
Zeit gefunden hat, den zarten Schimmer, von dem er umgeben war, zu
verdrängen.




20. Kapitel.

    In welchem Oliver Sikes überliefert wird.


Als Oliver am folgenden Morgen erwachte, war er nicht wenig verwundert,
ein Paar neue Schuhe mit starken, dicken Sohlen an der Stelle seiner
alten, sehr beschädigten zu erblicken. Anfangs freute er sich der
Entdeckung, weil er sie als eine Vorläuferin seiner Befreiung ansah;
allein er gab bald alle Gedanken dieser Art auf, als er sich allein mit
dem Juden zum Frühstück setzte, der ihm, und zwar auf eine Weise, die
ihn mit Unruhe erfüllte, sagte, daß er am Abend zu Bill Sikes gebracht
werden solle.

«Soll -- soll ich denn dort bleiben?» fragte Oliver angstvoll.

«Nein, nein, Kind, du sollst nicht dort bleiben», antwortete der Jude.
«Wir würden dich gar nicht gern missen. Sei ohne Furcht, Oliver; du
sollst wieder zurückkehren zu uns. Ha, ha, ha! Wir werden nicht sein
so grausam, dich fortzuschicken, mein Kind. Nein, beileibe nicht!» Der
alte Mann, der sich über das Feuer gebückt hatte und eine Brotschnitte
röstete, sah sich bei diesen spöttischen Worten um und kicherte, wie
um zu zeigen, er wisse es, daß Oliver gern entfliehen würde, wenn er
könnte.

«Ich glaube, Oliver,» fuhr er, die Blicke auf ihn heftend, fort, «du
möchtest wissen, weshalb du sollst zu Bill -- nicht wahr, mein Kind?»

Oliver verfärbte sich unwillkürlich, denn er gewahrte, daß der Jude in
seinem Innern gelesen, erwiderte indes dreist, daß er es allerdings zu
wissen wünsche.

«Nun, was meinst du wohl, weshalb?» fragte Fagin, der Antwort
ausweichend.

«Ich kann es nicht erraten, Sir», erwiderte Oliver.

«Pah! So warte, bis Bill dir's sagt», versetzte Fagin, sich mißvergnügt
abwendend, denn er hatte in Olivers Mienen wider Verhoffen nichts
entdeckt, nicht einmal Neugierde.

Die Wahrheit ist indessen, daß der Knabe allerdings sehr lebhaft zu
wissen wünschte, zu welchem Zwecke er Sikes überliefert werden sollte,
aber durch Fagins forschende Blicke und sein eigenes Nachsinnen zu sehr
außer Fassung geraten war, um für den Augenblick noch weitere Fragen
zu tun. Später fand sich keine Gelegenheit dazu, denn der Jude blieb
bis gegen Abend, da er sich zum Ausgehen anschickte, sehr mürrisch und
schweigsam.

«Du kannst brennen ein Licht», sagte er und stellte eine Kerze auf den
Tisch; «und da ist ein Buch, worin du kannst lesen, bis sie kommen,
dich abzuholen. Gute Nacht!»

«Gute Nacht, Sir», erwiderte Oliver schüchtern.

Der Jude ging nach der Tür und sah über die Schulter nach dem Knaben
zurück; dann stand er plötzlich still und rief ihn beim Namen.

Oliver blickte auf, der Jude wies nach dem Lichte hin und befahl ihm,
es anzuzünden. Oliver tat, wie ihm geheißen wurde, und sah, daß Fagin
mit gerunzelter Stirn aus dem dunkleren Teile des Gemachs forschend die
Augen auf ihn heftete.

«Hüte dich, Oliver, hüte dich!» sagte der Alte, warnend die rechte
Hand emporhebend. «Er ist ein brutaler Mann und achtet kein Blut, wenn
seins ist heiß. Was sich auch zuträgt, sprich kein Wort und tu', was
er dir sagt. Nimm dich in acht! -- wohl in acht!» Er hatte die letzten
Worte mit scharfer Betonung gesprochen, sein finsterer, drohender Blick
verwandelte sich in ein greuliches Lächeln, er nickte und ging.

Oliver legte den Kopf auf die Hand, als er allein war, und sann mit
pochendem Herzen den eben vernommenen Worten nach. Je länger er über
die Warnung des Juden nachdachte, in eine desto größere Ungewißheit
geriet er über ihren eigentlichen Sinn und Zweck. Er konnte sich nichts
Böses oder Unrechtes bei seiner Sendung zu Sikes denken, das nicht
ebensogut geschehen oder erreicht werden konnte, wenn er bei Fagin
blieb. Nach langem Nachsinnen kam er zu dem Schlusse, daß er ausersehen
sein möchte, Sikes als Aufwärter zu dienen, bis man einen besser dazu
geeigneten Knaben gefunden hätte. Er war zu sehr an Leiden und Dulden
gewöhnt und hatte zu viel gelitten in dem Hause, in welchem er sich
befand, als daß ihn die Aussicht auf eine Veränderung des Schauplatzes
seiner Widerwärtigkeiten sehr hätte betrüben können. Er blieb noch eine
Weile in Gedanken verloren, putzte seufzend das Licht und fing an in
dem Buche zu lesen, das ihm der Jude zurückgelassen.

Er hatte anfangs nur geblättert, allein eine Stelle erregte seine
Aufmerksamkeit im höchsten Grade, und bald las er um so eifriger. Das
Buch enthielt Erzählungen von berüchtigten Verbrechern aller Art und
trug auf jeder Seite die Spuren eines sehr häufigen Gebrauchs. Er las
hier von furchtbaren Verbrechen, die das Blut zu Eis erstarren ließen,
von Raubmorden, die auf offener Landstraße verübt worden waren, von
Leichen, die man vor den Augen der Menschen in den tiefen Brunnen
und Schächten verborgen hatte, ohne daß es jedoch gelungen wäre, sie
für die Dauer unten zu halten, so tief sie auch liegen mochten, und
zu verhüten, daß sie nach vielen Jahren ans Tageslicht kamen und die
Mörder durch ihren Anblick so sehr um alle Besinnung brachten, daß
sie ihre Schuld eingestanden und am Galgen ihr Leben endeten. Ferner
las er hier von Menschen, die in der Stille der Nacht in ihrem Bette
liegend von ihren eigenen bösen Gedanken zu so gräßlichen Mordtaten,
wie sie selbst sagten, aufgestachelt wurden, daß es einen kalt überlief
und einem die Glieder matt am Leibe niedersanken, wenn man es las. Die
fürchterlichen Beschreibungen waren so lebensgetreu und packend, daß
die schmutzigen Seiten ihm mit Blut bespritzt erschienen und die Worte,
die er las, in seinen Ohren widerhallten, als würden sie in hohlem
Murmeln von den Geistern der Ermordeten geflüstert.

In wahnsinniger Angst schloß Oliver endlich das Buch und schleuderte
es von sich, fiel auf die Knie nieder und flehte den Himmel an, ihn
vor solchen Untaten zu bewahren und ihn lieber sogleich sterben als
so fürchterliche Verbrechen begehen zu lassen. Er wurde allmählich
ruhiger und betete mit leiser, gebrochener Stimme um Errettung aus
den Gefahren, in welchen er sich befand, und, falls einem armen,
verstoßenen Knaben, der nie Elternliebe und Schutz gekannt, Beistand
und Hilfe aufgehoben wäre, daß sie ihm jetzt zuteil werden möchte, wo
er allein und verlassen von Schuld und Ruchlosigkeit umringt war.

Er lag noch, das Gesicht mit den Händen bedeckend, auf den Knien, als
ein Geräusch ihn aufschreckte. Er sah sich um, erblickte eine Gestalt
an der Tür und rief: «Wer ist da?»

«Ich -- ich bin es», erwiderte eine bebende Stimme.

Er hob das Licht empor und erkannte Nancy.

«Stell das Licht wieder auf den Tisch», sagte sie, das Gesicht
abwendend; «die Augen tun mir weh davon.»

Oliver sah, daß sie sehr blaß war, und fragte sie mitleidig, ob sie
krank wäre. Sie warf sich auf einen Stuhl, so daß sie ihm den Rücken
zukehrte, und rang die Hände, antwortete aber nicht.

«Gott verzeih' mir die Sünde!» rief sie nach einiger Zeit aus; «es ist
meine Absicht nicht gewesen -- ich habe nicht -- habe nicht von fern
daran gedacht!»

«Ist ein Unglück vorgefallen?» fragte Oliver. «Kann ich dir helfen?
Wenn ich es kann, so will ich's auch, gern, gern!»

Sie wiegte sich unter fortwährendem heftigen Händeringen hin und her,
faßte sich an die Kehle, als ob sie etwas würgte, und keuchte atemlos.

«Nancy!» rief Oliver bestürzt; «was ist dir denn?»

Sie schlug krampfhaft mit den Händen auf ihre Knie und mit den Füßen
auf den Boden und hüllte sich darauf schaudernd dicht in ihren Schal.
Oliver schürte das Feuer an, sie setzte sich an den Kamin, schwieg noch
eine Zeitlang, hob endlich den Kopf empor und blickte umher.

«Ich weiß nicht, wie mir bisweilen wird», sagte sie und stellte sich,
als wenn sie eifrig beschäftigt wäre, ihr Haar wieder zu ordnen; «ich
glaube, jetzt kommt's von der dumpfen Luft hier im Zimmer. Bist du
bereit, mit mir zu gehen, Nolly?»

«Soll ich mit dir fortgehen, Nancy?» fragte Oliver.

«Ja; ich komme von Bill Sikes», erwiderte sie. «Du sollst mit mir
gehen.»

«Wozu denn?» fragte Oliver zurückschreckend.

«Wozu?» wiederholte sie, schlug die Augen auf und wandte das Gesicht
ab, sobald sie Olivers Blicken begegneten. «Oh! zu nichts Bösem.»

«Das glaube ich dir nicht», sagte er. Er hatte sie genau beobachtet.

«So ist's gelogen, und glaub', was du willst», erwiderte sie und zwang
sich zu lachen. «Zu nichts Gutem also.»

Oliver entging es nicht, daß er einige Gewalt über Nancys bessere
Gefühle hatte, und wollte sich schon an ihr Mitleid mit seiner
hilflosen Lage wenden; allein es fiel ihm ein, daß es kaum elf Uhr
wäre, daß noch viele Leute auf den Straßen sein müßten, und daß ihm
ja wohl der eine oder andere Glauben schenken würde, wenn er ihn um
Beistand anspräche. Er trat vor, als ihm dieser Gedanke durch den Sinn
flog, und erklärte hastig und verwirrt, daß er bereit sei.

Nancy hatte ihn indes scharf im Auge behalten, erraten, was in seinem
Innern vorging, und ihr bedeutsamer Blick ließ ihn gewahren, daß sie
ihn durchschaut.

«Pst!» sagte sie, beugte sich herunter zu ihm, blickte vorsichtig
umher und wies nach der Tür. «Du kannst dir nicht helfen. Ich habe
mir deinetwegen alle mögliche Mühe gegeben, aber vergeblich. Du bist
umstellt und wirst scharf bewacht, und kannst du jemals loskommen, so
ist es jetzt nicht die Zeit.»

Sie war offenbar erregt, Oliver war davon betroffen und blickte ihr
sehr verwundert in das Gesicht. Sie schien die Wahrheit zu reden, war
blaß und zitterte heftig.

«Ich habe dich schon einmal vor übler Behandlung geschützt, will es
auch künftig tun und tue es jetzt», fuhr sie fort; «denn wenn ich
dich nicht holte, würden dich andere zu Sikes bringen, die viel
unglimpflicher mit dir umgehen möchten. Ich habe mich dafür verbürgt,
daß du ruhig und still sein würdest, und bist du es nicht, so wirst du
nur dir selbst und obendrein mir schaden, vielleicht an meinem Tode
schuld sein. Sieh hier! -- dies alles hab' ich für dich schon ertragen,
so wahr Gott sieht, daß ich's dir zeige.»

Sie wies ihm mehrere braune und blaue Streifen und Flecke an ihrer
Schulter und den Armen und sprach rasch weiter: «Denk daran, und laß
mich nicht eben jetzt noch mehr um deinetwillen leiden. Wenn ich dir
helfen könnte, würde ich's gern tun, ich habe aber die Macht nicht. Sie
wollen dir kein Leides zufügen, und was sie dich zwingen zu tun, ist
nicht deine Schuld. Pst! jedes Wort, was du sprichst, ist soviel als
ein Schlag für mich. Gib mir die Hand -- geschwind, deine Hand!»

Oliver reichte ihr mechanisch die Rechte, sie blies das Licht aus und
zog ihn nach. Die Haustür wurde rasch und leise von jemand geöffnet und
ebenso schnell hinter ihnen wieder verschlossen. Vor dem Hause stand
ein Mietswagen, sie schob ihn hinein und ließ die Fenster herunter.
Der Kutscher bedurfte keiner Weisung, sondern fuhr augenblicklich im
raschesten Trabe davon.

Nancy hielt fortwährend Olivers Hand fest und flüsterte ihm Trost,
Warnungen und Versprechungen in das Ohr. Alles war so überraschend
gekommen, daß er kaum Zeit hatte, seine Gedanken zu sammeln, als der
Wagen schon vor dem Hause hielt, in welchem der Jude am vergangenen
Abend Sikes aufgesucht hatte.

Einen einzigen kurzen Augenblick schaute Oliver umher, und ein
Hilferuf schwebte ihm auf den Lippen. Allein die Straße war öde und
menschenleer. Nancys bittende Stimme tönte in seinem Ohr, und während
er noch unentschlossen war, befand er sich schon im Hause und hörte
dasselbe sorgfältig verriegeln. Sikes trat mit einem Lichte oben an die
Treppe und begrüßte das Mädchen ungewöhnlich heiter und mild.

«Tyras ist mit Tom nach Hause gegangen», sagte er; «er würde im Wege
gewesen sein.»

«Das ist schön», erwiderte Nancy.

«Du hast ihn also?» bemerkte Sikes, als sie in das Zimmer eintraten.

«Ja, hier ist er.»

«Ging er ruhig mit?»

«Wie ein Lamm.»

«Freue mich, es zu hören,» sagte Sikes, Oliver finster anblickend,
«um seines jungen Leichnams willen. Komm her, Bursch, daß ich dir nur
gleich 'ne gute Lehre gebe, je eher, desto besser.»

Er setzte sich an den Tisch, und Oliver mußte sich ihm gegenüber
hinstellen.

«Weißt du, was dies ist?» fragte er, eine Taschenpistole zur Hand
nehmend.

Oliver bejahte.

«Dann schau hier. Dies ist Pulver, dies 'ne Kugel und das ein
Pfropfen.» -- Sikes lud die Pistole mit großer Sorgfalt und sagte, als
er fertig war: «Nun ist sie geladen.»

«Ja, Sir, ich sehe es», erwiderte Oliver bebend.

Sikes faßte die Hand des Knaben mit festem Griffe und setzte ihm den
Pistolenlauf an die Schläfe. Oliver konnte einen Angstschrei nicht
unterdrücken.

«Nun merk wohl, Bursch,» sagte Sikes, «sprichst du ein einziges Wort,
wenn du mit mir außer dem Hause bist, ausgenommen um zu antworten,
wenn ich dich frage, so hast du ohne weiteres die ganze Ladung im
Hirnkasten; also wenn du gesonnen sein solltest, ohne Erlaubnis zu
sprechen, so sag erst dein letztes Gebet her. Soviel ich weiß, wird
niemand besondere Nachforschung deinethalben anstellen, wenn dir der
Garaus gemacht ist; 's ist also bloß zu deinem Besten, daß ich mir so
viel Mühe gebe, dir ä Licht aufzustecken. Hast's gehört?»

Jetzt nahm Nancy das Wort und sagte sehr nachdrücklich und Oliver
etwas finster anblickend, wie um ihn aufzufordern, ihr so aufmerksam
als möglich zuzuhören: «Das Lange und Kurze von dem, was du sagen
willst, ist dies, Bill: Wenn er dich stört bei dem, was du vorhast, so
wirst du ihm, damit er nichts ausschwatzen kann, eine Kugel durch den
Kopf schießen und die Gefahr auf dich nehmen, dafür zu baumeln, wie du
diese Gefahr wegen sehr vieler anderer Dinge auf dich nimmst, die du im
Geschäft jede Woche deines Lebens tust.»

«Ganz recht!» bemerkte Sikes wohlgefällig. «Die Weibsen verstehen sich
drauf, alles mit den wenigsten Worten zu sagen, ausgenommen, wenn sie
zanken und schimpfen, wo sie's desto länger machen und die Worte nicht
sparen. Jetzo aber, nun er Bescheid weiß, schaff was zum Abendessen,
und dann wollen wir noch ä bissel dormen[AC], eh' wir losgehen.»

  [AC] schlafen.

Nancy gehorchte, deckte den Tisch und verschwand auf ein paar Minuten;
dann kehrte sie mit einem Krug Porter und einer Schüssel Hammelfleisch
zurück, die sie auf den Tisch stellte. Sikes aß und trank tüchtig und
warf sich auf das Bett, nachdem er Nancy geboten, ihn Punkt fünf Uhr zu
wecken, und Oliver, sich auf die Matratze neben seinem Bette zu legen.
Nancy schürte das Feuer und setzte sich an den Kamin, um die bestimmte
Zeit nicht zu verfehlen.

Oliver wachte noch lange und meinte, daß Nancy ihm vielleicht noch
ein paar Worte zuflüstern würde; allein sie regte sich nicht, und er
schlief endlich ein.

Als er erwachte, stand Teegeschirr auf dem Tische; Sikes steckte
verschiedene Sachen in die Taschen seines über einer Stuhllehne
hängenden Überrocks, und Nancy war beschäftigt, das Frühstück zu
bereiten. Der Tag war noch nicht angebrochen, und das Licht brannte
noch; der Regen schlug gegen die Fenster, und der Himmel sah schwarz
und wolkig aus.

Sikes trieb Oliver zur Eile an, der hastig sein Frühstück einnahm,
worauf ihm Nancy ein Halstuch umband; Sikes hing ihm einen großen,
groben Mantel über die Schultern, faßte ihn bei der Hand, zeigte ihm
den Kolben der Pistole und ging mit ihm fort, nachdem er sich von Nancy
verabschiedet hatte.

Oliver drehte sich an der Tür um, in der Hoffnung, einen Blick von
Nancy zu erhalten, die sich jedoch schon wieder an den Kamin gesetzt
hatte und regungslos in das Feuer schaute.




21. Kapitel.

    Der Aufbruch.


Es war ein unfreundlicher Morgen, als sie auf die Straße hinaustraten.
Es ging ein scharfer Wind, und es regnete stark. Auf der Straße standen
große Pfützen, und die Rinnsteine waren überfüllt. Am Himmel zeigte
sich ein schwacher Schimmer des kommenden Tages, der aber das Düstere
der Szene eher verstärkte als verminderte, da das trübe Licht nur dazu
diente, das der Straßenlaternen zu dämpfen, ohne einen wärmeren oder
lichteren Farbenton in das Grau der nassen Dächer und schmutzigen
Straßen zu bringen. Es schien noch niemand in diesem Stadtviertel
aufgestanden zu sein; die Fensterläden der Häuser waren noch fest
verschlossen, und niemand ließ sich auf den öden, schmutzigen Straßen
blicken.

Der Tag brach erst an, als sie sich Bethnal Green näherten. Viele
Laternen waren schon gelöscht; dann und wann fuhr ein Marktwagen
langsam daher, oder es rollte eine Postkutsche vorüber. Die Schenken
standen schon offen und waren hell erleuchtet. Allmählich begannen
sich auch einige Läden zu öffnen. Hier kamen Gruppen von Arbeitern,
die zur Werkstatt oder Fabrik gingen, dort Männer und Frauen
mit Fischkörben auf dem Kopfe, mit Gemüse beladene Eselkarren,
Fleischerwagen mit geschlachtetem Vieh, Milchfrauen mit ihren Kannen
-- ein ununterbrochener Menschenstrom, der zu Fuß oder zu Wagen in die
östlichen Vorstädte hineinflutete. Als sie sich der City näherten,
wurde der Lärm und der Verkehr immer stärker; als sie die Straßen
zwischen Shoreditch und Smithfield durchschritten, war er zu einem
sinnbetäubenden Gewühl angeschwollen. In Shoreditch und Smithfield
war lautes Getümmel und Gedränge. Es war Markttag. Oliver war vor
Erstaunen außer sich. Er meinte, ganz London wäre aus einer ganz
besonderen Veranlassung in Bewegung. Welch eine Geschäftigkeit, welch
ein Gewühl, Rufen, Lärmen, Zanken und Streiten -- jeden Augenblick neue
Gegenstände, neue Gesichter und Menschenknäuel.

Sikes zog seinen Begleiter rastlos fort, beachtete kaum, was Oliver die
Sinne verwirrte, nickte nur dann und wann einem begegnenden Bekannten
einen Gruß zu und lenkte nach Holborn ein. Er trieb zur Eile an, Oliver
vermochte, fast atemlos, kaum Schritt mit ihm zu halten und wurde so
rasch fortgerissen, daß es ihm fast war, als wenn er über die Erde
dahinschwebte. Auf der Straße nach Kensington hielt Sikes einen leeren
Karren an und forderte den Eigentümer auf, ihn selbst und seinen Knaben
bis Isleworth mitzunehmen. Er war mit dem Kärrner bald einig und hob
Oliver in den Karren, wobei er nicht vergaß, bedeutsam auf seine
Rocktasche zu schlagen.

Nachdem sie durch Kensington, Hammersmith, Chiswick, Kew Bridge und
Brentford gefahren waren, ließ Sikes halten, stieg mit Oliver aus,
wartete, bis der Fuhrmann vollständig aus seinem Gesichtskreise
verschwunden war, und setzte dann mit Oliver seine Wanderung fort. Sie
wandten sich erst nach links, dann nach rechts und kamen an vielen
großen Gärten und schönen Villen vorüber, kehrten aber die ganze
Zeit über nur einmal ein, um einen Schluck Bier zu trinken. Endlich
erreichten sie eine Stadt, und Oliver sah an der Wand eines Hauses mit
großen Buchstaben den Namen «Hampton» geschrieben. Sie warteten einige
Stunden zwischen den Feldern und wandten sich dann nach der Stadt
zurück; Sikes kehrte in einem alten, verfallenen Wirtshause mit einem
verblichenen Schilde ein und bestellte ein Mittagessen beim Küchenfeuer.

Die Küche war ein alter, niedriger Raum mit einem großen Balken quer
über die Decke und hochlehnigen Bänken in der Nähe des Feuers, auf
denen mehrere rauh aussehende Männer rauchend und trinkend saßen. Sie
beachteten Oliver gar nicht und Sikes sehr wenig, und da letzterer
ebenfalls wenig Notiz von ihnen nahm, so saß er mit seinem kleinen
Gefährten ganz allein in einer Ecke, ohne sich durch ihre Anwesenheit
im geringsten stören zu lassen.

Sie aßen etwas kaltes Fleisch und blieben so lange sitzen, während
Sikes sich den Genuß von drei bis vier Pfeifen gönnte, daß Oliver ganz
sicher glaubte, sie würden heute nicht weitergehen. Da er von der
weiten Wanderung ermüdet und so früh aufgestanden war, so wurde er
schläfrig und versank endlich, überwältigt von den Strapazen und dem
Tabakrauch, in tiefen Schlummer.

Es war schon ganz dunkel, als er durch einen Rippenstoß, den ihm Sikes
versetzt hatte, geweckt wurde. Als er sich genügend ermuntert hatte, um
aufrecht sitzen und sich umschauen zu können, sah er seinen würdigen
Begleiter mit einem Fuhrmann, der ziemlich betrunken zu sein schien
und nach Shepperton wollte, bei einem Glase Ale zusammensitzen und
hörte, wie er ihn fragte, ob er ihn und den Knaben mitnehmen wolle.
Der Fuhrmann willigte ein, und als es Zeit zum Abfahren war, hob Sikes
Oliver in den Wagen, der sich sofort in Bewegung setzte und in scharfem
Trabe aus der Stadt rasselte.

Der Abend war sehr dunkel. Ein dichter Nebel stieg von dem Flusse und
dem moorigen Boden ringsherum auf und breitete sich über die nassen
Felder aus. Dazu war es schneidend kalt; alles war düster und schwarz.
Kein Wort wurde gesprochen; denn der Fuhrmann war schläfrig geworden,
und Sikes befand sich nicht in der Stimmung, ein Gespräch mit ihm
anzuknüpfen. Oliver saß zusammengekauert in einer Ecke des offenen
Wagens, von Unruhe und Angst gepeinigt, und die riesigen Bäume, die wie
in wilder Freude über die Trostlosigkeit der Gegend ihre Zweige heftig
hin und her bewegten, kamen ihm wie gespenstische Wesen vor.

Als sie an der Kirche in Sunbury vorüberfuhren, schlug es sieben
Uhr. Sie mochten noch zwei oder drei Meilen gefahren sein, als Sikes
abstieg und, Oliver bei der Hand fassend, weiterging. Er kehrte, was
der müde Knabe erwartet hatte, in Shepperton nicht ein, sondern ging
durch den Schlamm, die Finsternis und düstere Gassen und über öde,
offene Plätze weiter, bis sich die Lichter einer Stadt in geringer
Entfernung zeigten. Sie gelangten an eine Brücke, und Sikes lenkte in
einen Uferweg ein. Oliver erschrak heftig; er glaubte, daß Sikes ihn an
diesen einsamen Ort gebracht hätte, um ihn zu ermorden. Er wollte sich
schon niederwerfen, um verzweifelt für sein junges Leben zu kämpfen,
als sie vor einem einzelnen, verfallenen Hause standen. Licht war darin
nicht sichtbar; es schien unbewohnt zu sein. Sikes trat leise an die
Tür, legte die Hand auf den Griff, und beide standen auf dem dunklen
Hausflur.




22. Kapitel.

    Der Einbruch.


«Wer da?» rief eine laute heisere Stimme.

«Mach keinen solchen Hamore[AD]», sagte Sikes, während er die Tür
verriegelte. «Ä Chandel[AE], Toby!»

  [AD] Lärm.

  [AE] Licht.

«Aha, mein guter Chawwer», erwiderte dieselbe Stimme; «ä Chandel,
Barney, ä Chandel! Führ den Herrn 'nein, Barney; wach aber erst auf,
wenn dir's recht ist.»

Man hörte, daß irgend etwas Gewichtiges nach jemand geworfen wurde und
sodann auf die Erde fiel.

«Hörst nicht?» rief dieselbe Stimme. «Da steht Bill Sikes draußen im
Dunkeln, und du dormst, als wenn du 'nen Schlaftrunk g'soffen hätt'st
und nichts Stärkeres. Wirst du jetzt munter, oder soll ich dich mit'm
eisernen Leuchter wecken?»

Endlich schlürfte der Kellner im Hotel von Saffron Hill mit Licht
heran und begrüßte Sikes mit wirklicher oder erkünstelter Freude.
Sikes stieß Oliver voran in ein niedriges, düsteres Gemach mit
einigen gebrechlichen Stühlen, einem Tische und einem sehr schlechten
Bette, auf welchem ein Mann ausgestreckt und aus einer langen
Tonpfeife rauchend lag. Er trug einen dunkelbraunen Rock mit großen
Metallknöpfen, ein orangefarbenes Halstuch, eine buntfarbige Weste und
hellbraune Beinkleider. Mr. Crackit (denn er war es) hatte dünnes,
rötliches, in Locken gedrehtes Haar, durch das er von Zeit zu Zeit
mit schmutzigen, beringten Fingern hindurchfuhr. Er war etwas über
Mittelgröße, und seine Beine schienen ziemlich dünn zu sein, wodurch
indes keineswegs die Bewunderung und Zufriedenheit vermindert wurde,
womit er oft genug seine hohen Stiefel beäugelte. «Bill, geliebter
Freund,» rief er Sikes entgegen, «ich freue mich, dich zu sehen.
Fürchtete fast schon, daß du's aufgegeben hätt'st, in welchem Fall
ich's auf meine eigene Faust versucht haben würde. Was der Teufel!»
setzte er, als er Oliver erblickte, erstaunt hinzu, richtete sich zum
Sitzen empor und fragte, wer der Knabe wäre.

«Nun, 's ist eben der Knabe», erwiderte Sikes und setzte sich an den
Kamin.

«Einer von Fagin seinen», bemerkte Barney grinsend.

«Von Fagin, so?» rief Toby, nach Oliver hinblickend, aus. «Was für'n
prachtvoller Junge er werden wird für die Taschen der alten Damen in
Kirchen und Kapellen! Sein Ponum[AF] ist so gut wie 'n Kap'tal für ihn.»

  [AF] Mund, Gesicht.

«So schweig doch still -- 's ist schon mehr als zuviel Schwätzens
davon», unterbrach ihn Sikes ungeduldig und flüsterte ihm etwas in
das Ohr. Toby Crackit lachte ausgelassen und starrte Oliver lange
verwundert an.

«Gebt uns zu essen und zu trinken -- es wird uns Courage machen -- mir
wenigstens», sagte Sikes. «Setz dich ans Feuer, Bursch, und ruh dich
aus, du gehst noch mit uns aus heute nacht, wenn auch eben nicht weit.»

Oliver sah ihn in stummer und furchtsamer Verwunderung an, setzte
sich ans Feuer und stützte, kaum wissend, was um ihn her und mit ihm
vorging, den schmerzenden Kopf auf die Hände. Der jüdische Jüngling
trug Speisen und Getränk auf, und Toby und Sikes tranken auf ein
glückliches Schränken. Toby füllte ein Glas, reichte es Oliver und
forderte ihn auf, es auszutrinken. Der Knabe versicherte, nicht trinken
zu können, und bat mit jammervollen Mienen, ihn damit zu verschonen.
Toby ließ sich jedoch nicht abweisen.

«Hinunter damit!» rief er. «Meinst du, ich wüßte nicht, was dir gut
ist? Bill, sag's ihm, daß er trinkt!»

«Soll ich dich lehren, gehorsam zu sein?» sagte Sikes, die Hand in
die Tasche steckend. «Gott verdamm' mich, wenn mir der Bube nicht
mehr Beschwerde macht, als ein ganz Dutzend Baldowerer. Trink aus,
Galgenstrick, oder --!»

Erschreckt durch die drohenden Gebärden der beiden Männer, stürzte
Oliver hastig den Inhalt des Glases hinunter und wurde sofort von
einem heftigen Husten befallen, worüber Toby und Barney in ein lautes
Gelächter ausbrachen und sogar der grämliche Sikes den Mund verzog.

Nachdem Sikes seinen Hunger gestillt hatte (Oliver konnte außer einer
Rinde Brot, die er zu essen gezwungen wurde, nichts zu sich nehmen),
legten sich die beiden Männer zu einem kurzen Schlafe nieder. Oliver
blieb auf seinem Stuhle am Feuer sitzen, und Barney streckte sich, in
eine Decke gehüllt, dicht neben dem Kamine aus.

Sie schliefen oder schienen zu schlafen, denn es regte sich niemand
außer Barney, der ein paarmal aufstand, um Kohlen in das Feuer zu
werfen. Oliver verfiel in einen dumpfen Schlummer, der von schweren,
ängstlichen Träumen beunruhigt wurde, bis Toby aufsprang und erklärte,
es sei halb zwei Uhr. Im nächsten Augenblicke waren die beiden
anderen auf den Beinen, und alle drei waren eifrig dabei, die nötigen
Vorbereitungen zu treffen. Die beiden Schränker zogen sogleich ihre
Überröcke an und verhüllten sich mit Tüchern bis über den Mund. Barney
füllte eiligst ihre Taschen mit mehreren Gegenständen an, die er aus
einem Schranke holte.

«Barney, meine Lupperts[AG]», sagte Toby Crackit.

  [AG] Pistolen.

«Da sind sie. Ihr habt sie selbst geladen.»

«Ja, ja. Die Wurmer[AH].»

  [AH] Bohrer.

«Die hab' ich», fiel Sikes ein.

«Chlamones[AI], Drehbarsel, Hänenehres[AJ], nichts vergessen?» fragte
Toby, ein kleines Brecheisen einsteckend.

  [AI] Diebesschlüssel.

  [AJ] Laterne.

«Alles da», antwortete Sikes. «Barney, die grandige Makel. Es ist
höchste Zeit.»

Barney reichte ihm und Toby große Knotenstöcke und legte Oliver den
Mantel um.

«Jetzt also», sagte Sikes, seine Hand ausstreckend.

Oliver, der durch die ungewohnte Anstrengung, die schlechte Luft und
das ihm aufgezwungene Getränk völlig betäubt war, legte seine Hand
mechanisch in die Sikes'.

«Nimm seine andere Hand, Toby», sagte Sikes. «Schau nach, Barney, ob
alles sicher ist.»

Der Kellner ging vor die Tür und kehrte mit der Meldung zurück, daß
alles still sei. Die beiden Schränker eilten hinaus und zogen Oliver
mit sich fort.

Die Nacht war rabenschwarz und der Nebel so dicht, daß nach wenigen
Minuten große Tropfen an Olivers Augenbrauen hingen. Sie eilten im
tiefsten Schweigen über die Brücke und durch den nächstgelegenen Ort
und erreichten um zwei Uhr ein einzeln stehendes, von einer Mauer
umgebenes Haus, die Toby Crackit sogleich erklomm. Sikes hob Oliver
empor, und nach wenigen Augenblicken waren alle drei hinüber. Sikes
und Toby schlichen nach dem Hause und zogen den Knaben mit sich fort,
dem die Sinne fast entschwanden, denn jetzt zum erstenmal tauchte der
Gedanke in ihm auf, daß Sikes auf Raub, wo nicht auf Mord ausginge
und ihn als Werkzeug dabei zu gebrauchen denke. Er schlug die Hände
zusammen, und seinen Lippen entfloh ein unwillkürlicher Schrei des
Entsetzens. Ihm schwindelte, kalter Schweiß stand auf seiner Stirn, er
wankte und fiel auf die Knie nieder.

«Steh auf!» flüsterte Sikes und zog bebend vor Wut die Pistole aus der
Tasche; «steh auf, oder ich schieße dir den Brägen aus'm Kopfe 'raus!»

«Oh, um Gottes willen, lassen Sie mich gehen!» rief Oliver; «lassen
Sie mich fortlaufen und hinter dem Zaune sterben. Ich will nie wieder
nach London kommen -- nie, nie; haben Sie Barmherzigkeit mit mir und
zwingen Sie mich nicht, zu stehlen. Um der Liebe der Engel willen, die
im Himmel wohnen, haben Sie Erbarmen mit mir!»

Sikes stieß einen fürchterlichen Fluch aus und spannte den Hahn, Toby
schob indes seine Hand zur Seite, hielt Oliver den Mund zu und zog ihn
fort nach dem Hause.

«Pst!» flüsterte er; «das ist hier nichts. Ist's nicht anders und
soll's sein, so sprich ein Wort, und ich schlag' ihn auf den Kopf,
was ebensogut ist und kein Geräusch macht. Hierher, Bill, brich den
Fensterladen auf. Ich stehe dafür, er hat jetzt Courage genug. Ich
hab's g'sehn, daß ältere als er in 'ner kalten Nacht 's Kanonenfieber
auf 'ne Minute oder so was g'habt haben.»

Sikes murmelte Verwünschungen gegen Fagin, ihm Oliver zu einem solchen
Unternehmen geschickt zu haben, setzte das Brecheisen an, und nach
kurzer Zeit war der Fensterladen geöffnet. Das kleine Gitterfenster war
fünf bis sechs Fuß über der Erde im Hinterhause und gehörte zu einem
kleinen, zum Waschen oder Brauen bestimmten Gemach am unteren Ende des
Hausflurs. Das Gitter war gleichfalls bald durchbrochen.

«Jetzt hör' und merk, du kleiner Teufelsbraten!» flüsterte Sikes, zog
eine Blendlaterne aus der Tasche und hielt sie Oliver gerade vor das
Gesicht. «Ich stecke dich durch dies Fenster hier. Nimm diese Laterne,
geh leise die Stufen gerade vor dir 'nauf über den Flur nach der
Haustür, mach' sie auf und laß uns ein. -- Ist die Waschhaustür offen,
Toby?»

Toby antwortete, nachdem er hineingesehen hatte: «Sie steht weit offen,
und sie lassen sie immer offen, daß der Hund, der hier sein Lager hat,
im Hause 'rumspazieren kann. Ha, ha, ha! Wie hübsch ihn Barney gestern
abend weggelockt hat!»

So leise Crackit gesprochen und gekichert hatte, befahl ihm doch Sikes
in gebieterischem Tone, still zu schweigen und an das Werk zu gehen.
Er setzte die Laterne auf die Erde, stellte sich unter das Fenster,
die Hände auf die Knie gestützt, mit dem Kopfe gegen die Wand, Sikes
stieg auf den Rücken Tobys und hob Oliver durch das Fenster in das Haus
hinein.

«Nimm die Leuchte», flüsterte er ihm zu. «Siehst du die Stufen da vor
dir?»

Oliver keuchte, mehr tot als lebendig, ein mattes «Ja». Sikes wies mit
der Pistole nach der Haustür hin und erinnerte ihn, daß er ihn bis zur
Haustür fortwährend in Schußweite hätte und ihn niederschießen würde,
wenn er sich verweilte oder auch nur einen Schritt zur Seite ginge.

«'s ist in 'ner Minute geschehen», flüsterte er Oliver zu. «Sobald ich
dich loslasse, tu, was dir geheißen ist. Pst!»

«Was ist denn?» fragte Toby.

Sie horchten.

«Nichts», sagte Sikes und ließ Oliver los. «Jetzt vorwärts!»

Der Knabe hatte sich indes wieder einigermaßen gesammelt und den
raschen und festen Entschluß gefaßt, wenn es auch sein Tod wäre, den
Versuch zu machen, auf dem Hausflur zur Seite zu springen und Lärm zu
machen. Von diesem Gedanken erfüllt, ging er bebend vorwärts.

«Komm zurück!» schrie Sikes plötzlich laut; «zurück, zurück!»

Erschreckt durch die plötzliche Unterbrechung der Totenstille und ein
lautes Geschrei, ließ Oliver die Laterne fallen und stand still, ohne
zu wissen, ob er vorwärts gehen oder entfliehen sollte. Das Geschrei
wiederholte sich, es zeigte sich ein Licht -- es war ihm, als sähe er
bestürzte, halb angekleidete Männer an der Tür -- es schwamm ihm vor
den Augen -- ein Gewehr blitzte auf -- ein Donner traf sein Ohr -- er
taumelte zurück. Sikes faßte ihn sogleich beim Kragen, feuerte nach den
zurückweichenden Männern und zog ihn durch das Fenster.

«Drück den Arm dichter an den Leib», flüsterte er, während er ihn
durchzog. «Toby, ein Tuch! Sie haben ihn getroffen. Geschwind! Höll'
und Teufel, wie der Bursch blutet!»

Oliver war sich dunkel bewußt, daß der Lärm im Hause immer mehr zunahm
und daß er rasch fortgetragen wurde. Das Geräusch verlor sich in der
Ferne, die Sinne entschwanden ihm gänzlich, es war ihm, als wenn eine
kalte Hand sein Herz umfaßte, es schlug, und er sah und hörte nichts
mehr.




23. Kapitel.

    Welches das Wesentliche einer anmutigen Unterredung zwischen Mr.
    Bumble und einer Dame enthält und zugleich dartut, daß sogar ein
    Kirchspieldiener in einigen Punkten empfänglich sein kann.


Der Abend war bitter kalt, und ein heftiger, schneidender Wind
trieb dichte Schneewirbel durch die Luft. Es war ein Abend für die
Wohlbehäbigen, beim lustigen, prasselnden Feuer Gott zu danken, daß
sie daheim waren, und für die heimatlosen Elenden und Hungrigen, sich
niederzulegen und zu sterben. Ach! viele solcher Auswürflinge der
Gesellschaft schließen zu solchen Stunden die Augen auf unseren öden,
verlassenen Straßen, und sie können dieselben, was auch ihr Verbrechen
gewesen sein mag, kaum in einer schlimmeren Welt wieder öffnen.

So sah es draußen aus, als Mrs. Corney, die Vorsteherin des
Armenhauses, in welchem Oliver Twist das Licht der Welt erblickt
hatte, sich in ihrem kleinen Zimmer an ihren behaglichen Kamin setzte
und wohlgefällig ihren kleinen runden Teetisch überblickte, und als
sie gar von dem Tische nach der Feuerstelle hinsah, wo der denkbar
kleinste aller Kessel ein leises Lied mit leiser Stimme sang, wuchs
augenscheinlich ihre innere Befriedigung, und zwar in einem solchen
Grade, daß Mrs. Corney lächelte.

«Ja,» sagte sie, indem sie ihren Arm auf den Tisch stützte und sinnend
ins Feuer blickte, «ich bin überzeugt, wir haben alle volle Ursache,
dankbar zu sein. Volle Ursache, wenn wir es nur anerkennen wollten.»

Sie schüttelte betrübt den Kopf, als wenn sie die geistige Blindheit
der Armen beklagte, die es nicht erkannten, und fing an, ihren Tee zu
bereiten, indem sie mit ihrem silbernen Löffel (Privateigentum!) tief
in eine zinnerne Teebüchse fuhr.

Wie geringe Dinge das Gleichgewicht unserer schwachen Gemüter stören
können! Der schwarze Teetopf war sehr klein und leicht gefüllt, das
Wasser lief über und verbrannte ein wenig ihre Hand.

«Oh, über den verwünschten Topf!» sagte sie, ihn hastig aus der Hand
setzend. «Das kleine dumme Ding hält nur ein paar Tassen. Wem ist er
nütze -- ausgenommen einer armen, einsamen, verlassenen Frau, wie ich
es bin! Ach, ach!»

Bei diesen Worten sank die würdige Dame auf ihren Stuhl und dachte,
abermals den Arm auf den Tisch gestützt, über ihr Geschick nach. Der
kleine Topf und die einzelne Tasse hatten traurige Erinnerungen an
Mr. Corney (der noch nicht länger als fünfundzwanzig Jahre tot war)
erweckt. Sie war davon ganz überwältigt.

«Ich bekomme niemals einen anderen,» sagte sie kummervoll und mißmutig;
«bekomme niemals einen anderen -- wie ihn!»

Wir können nicht entscheiden, ob sich dieser Stoßseufzer auf ihren
Seligen oder den Teetopf bezog, auf welchen zum wenigsten ihre Blicke
gerichtet waren, und der also auch gemeint sein konnte. Sie hatte kaum
die erste Tasse gekostet, als leise geklopft wurde.

«Herein!» rief Mrs. Corney ärgerlich. «Sicher will eins der alten
Weiber sterben. Sie sterben immer, wenn ich bei Tisch sitze oder meine
Tasse Tee trinke. Bleiben Sie nicht da draußen stehen; Sie lassen sonst
die kalte Zugluft herein. Was ist denn schon wieder los?»

«Nichts, Ma'am, nichts», antwortete eine Männerstimme.

«Himmel! sind Sie es wirklich, Mr. Bumble?» rief die Dame jetzt weit
freundlicher aus.

«Zu Diensten, Ma'am», erwiderte Bumble, der draußen stehengeblieben
war, um seine Schuhe zu reinigen und den Schnee von seinem Hute zu
schütteln, und der jetzt eintrat, in der einen Hand seinen dreieckigen
Hut und in der anderen ein Bündel. «Darf ich die Tür schließen, Ma'am?»

Mrs. Corney zögerte verschämt, zu antworten, weil es als eine
Ungeschicklichkeit angesehen werden konnte, wenn sie mit Mr. Bumble bei
geschlossener Tür eine Unterredung unter vier Augen hätte, und Bumble
benutzte die Zögerung, um die Tür ohne erhaltene Erlaubnis zu schließen.

«Schlechtes Wetter, Mr. Bumble», bemerkte die Matrone.

«Ja, ja, Ma'am,» sagte Bumble, «schlechte Witterung für das Kirchspiel.
Wir haben heute nachmittag zwanzig Brote und anderthalb Käse
weggegeben, und das Armenpack ist doch nicht zufrieden. Da ist ein
Mann, der in Anbetracht seiner Frau und einer zahlreichen Familie ein
großes Brot und ein ganzes Pfund Käse erhielt, und bedankte er sich,
bedankte er sich wohl? Prosit die Mahlzeit! Er bettelte obendrein
um Kohlen, und wenn's auch nur ein Taschentuch voll wäre, sagte er.
Und was wollte er mit den Kohlen? Seine Käse darüber rösten und dann
wiederkommen und um noch mehr betteln! So machen sie's, Ma'am -- so
machen sie's alle. Geben Sie ihnen eine Schürze voll Kohlen, und sie
werden übermorgen wiederkommen und eine neue haben wollen -- die
Frechdachse! Vorgestern kam ein Mann, der kaum einen Fetzen auf seinem
Leibe hatte (hier schlug Mrs. Corney verschämt die Augen nieder) --
Sie sind verheiratet gewesen, Ma'am, und so kann ich's wohl sagen --
in des Direktors Haus, als der Herr gerade eine Mittagsgesellschaft
hatte, und bat um Unterstützung. Da er nicht fortgehen wollte und die
Gesellschaft belästigte, ließ ihm der Direktor ein Pfund Kartoffeln
und ein Maß Hafermehl reichen. >Mein Gott,< sagte der undankbare
Bösewicht, >was soll ich damit? Sie könnten mir ebensogut 'ne eiserne
Brille geben.< -- >Sehr wohl,< erwiderte ihm der Direktor, die Spende
wieder an sich nehmend. >Ihr werdet hier sonst nichts bekommen.< --
>Dann sterb' ich auf der offenen Straße<, sagte der Landstreicher. >Das
werdet Ihr wohl bleiben lassen<, sagte der Direktor. Der Bettler ging
und starb auf der Straße. Was sagen Sie zu 'nem solchen Eigensinne,
Mrs. Corney?»

«Es übersteigt alle Begriffe», versetzte die Dame. «Aber halten Sie als
ein Mann von Erfahrung die Unterstützungen außerhalb des Armenhauses
nicht für sehr nachteilig, Mr. Bumble?»

«Mrs. Corney,» erwiderte der Kirchspieldiener mit dem Lächeln bewußter
Überlegenheit, «es ruht vielmehr in ihnen des Kirchspiels Schutz
und Sicherheit. Ihr großes Prinzipium ist, den Armen just das zu
geben, dessen sie nicht bedürfen; sie werden es dann überdrüssig,
wiederzukommen. Deshalb, Mrs. Corney, ist in den impertinenten
Zeitungen so oft die Rede davon, daß arme Kranke mit Käse unterstützt
würden, was jetzt im ganzen Lande die Regel ist. Dies sind jedoch
Dienstgeheimnisse, wovon zu reden jedermann verboten sein sollte,
ausgenommen uns Kirchspielbeamten. Mrs. Corney,» fügte Bumble, sein
Bündel öffnend, hinzu, «dies ist echter Portwein von bester Qualität,
den das Kollegium für die Kranken abzuziehen befohlen hat.»

Er stellte die beiden mitgebrachten Flaschen auf die Kommode, steckte
sein Tuch bedächtig in die Tasche und schickte sich zum Fortgehen an.
Die mitleidige Dame bemerkte, es wäre recht kaltes Wetter, und fragte
ihn schüchtern, ob ihm nicht beliebe, ein Schälchen Tee anzunehmen. Er
legte sogleich den Hut wieder aus der Hand, nahm an dem kleinen, runden
Tische Platz, lächelte und blickte Mrs. Corney so zärtlich an, daß sie
verlegen wegsehen und den Teekessel anblicken mußte. Sie schenkte ihm
ein, er breitete sein Taschentuch über die Knie und fing an zu trinken
und zu essen, seinen Genuß von Zeit zu Zeit mit einem tiefen Seufzer
begleitend, was jedoch seinem Appetit keineswegs schadete, sondern
denselben vielmehr zu stärken schien.

«Ich sehe, Ma'am,» sagte er nach ziemlich geraumer Zeit, «Sie haben
eine Katze und auch kleine Kätzchen.»

«Sie glauben gar nicht, wie lieb ich sie habe, und wie vergnügt und
lustig sie bei mir sind, Mr. Bumble.»

«Mrs. Corney, ich muß sagen: jede Katze, die bei Ihnen und täglich um
Sie wäre und Sie nicht lieb hätte, müßte ein Esel sein.»

«Ah, Mr. Bumble!»

«Es ist die Wahrheit, und ich würde sie mit Vergnügen ersäufen.»

«Mr. Bumble, was Sie für ein hartherziger Mann sind!»

«Ein hartherziger Mann?» wiederholte Bumble mit einem zärtlichen
Seufzer, ergriff und drückte Mrs. Corneys kleinen Finger, rückte ein
wenig um den Tisch herum und rückte immer näher, bis sein Stuhl dicht
neben dem Stuhle Mrs. Corneys stand, die nicht fortrücken konnte, weil
sie sonst dem Kamin zu nahe gekommen sein würde, was zwischen den
beiden Feuern die noch gefährlichere Nähe war. Rechts konnten ihre
Kleider Feuer fangen, links nur ihr Herz; rechts konnte sie auf den
Rost, links nur in Mr. Bumbles Arme fallen. Sie war eine kluge und
umsichtige Frau, berechnete ohne Zweifel die möglichen Folgen, blieb
ganz still sitzen und schenkte Mr. Bumble noch eine Tasse Tee ein.

«Ein hartherziger Mann, Mrs. Corney?» sagte Bumble, seinen Tee
umrührend und ihr in das Angesicht schauend; «sind Sie eine hartherzige
Frau?»

«Mein Gott! Was für eine Frage für einen unverheirateten Mann!» rief
die Matrone aus. «Was wollen Sie damit sagen, Mr. Bumble?»

Bumble trank bis auf den letzten Tropfen aus, verspeiste eine geröstete
Butterschnitte, entfernte die Krumen von seinen Knien, wischte sich
die Lippen und küßte die Matrone bedächtig.

«Mr. Bumble!» rief die keusche Dame flüsternd; denn ihr Schrecken war
so groß, daß ihr die Stimme fast versagte: «Mr. Bumble, ich werde
schreien!»

Bumble sagte gar nichts, sondern legte langsam und mit Würde den Arm um
ihren Leib. Da sie die Absicht, schreien zu wollen, bereits angekündigt
hatte, so würde sie bei dieser neuen Keckheit natürlich geschrien
haben; allein es wurde unnötig, indem hastig an die Tür geklopft wurde,
worauf Bumble ebenso eilig aufsprang und mit großer Vehemenz die
Portweinflaschen abzustäuben anfing. Mrs. Corney rief: «Herein!» Eine
alte Frau steckte den Kopf in das Zimmer und verkündete, daß die alte
Sarah im Sterben läge.

«Was geht es mich an!» entgegnete Mrs. Corney verdrießlich. «Kann ich
sie am Leben erhalten?»

«Das kann freilich niemand, Ma'am; ihr ist nicht mehr zu helfen. Ich
habe viel Kranke sterben sehen, kleine Kinder wie Männer in ihren
besten Jahren, und weiß es auf ein Haar, wann der Tod im Anzuge ist.
Jedoch ist sie unruhig in ihrem Geist und sagt, daß sie Ihnen noch
etwas Notwendiges anzuvertrauen hätte. Sie könnte nicht ruhig sterben,
eh' Sie nicht bei ihr gewesen wären, Ma'am.»

Die würdige Matrone murmelte eine beträchtliche Anzahl von
Verwünschungen gegen die alten Frauen, die niemals sterben könnten,
ohne absichtlich ihre Vorgesetzten zu belästigen, hüllte sich in einen
wärmenden Mantel, bat Bumble, zu bleiben, bis sie wieder da wäre, und
entfernte sich verdrießlich und keifend mit der an sie abgeschickten
alten Frau.

Was Mr. Bumble tat, als er sich allein sah, war etwas unerklärlich. Er
öffnete nämlich den Schrank, zählte die Teelöffel, wog die Zuckerzange,
prüfte einen Milchgießer, ob er auch von echtem Silber wäre, setzte,
nachdem er seine Wißbegier befriedigt, den dreieckigen Hut auf und fing
an, sehr gravitätisch im Zimmer umherzutanzen, nahm darauf den Hut
wieder ab, setzte sich an den Kamin, blickte umher und nahm offenbar im
Geist ein Inventar über die im Zimmer befindlichen Mobilien auf.




24. Kapitel.

    Welches sehr kurz ist, aber doch für wichtig befunden werden könnte.


Die Alte, welche die Ruhe des Zimmers Mrs. Corneys gestört hatte, war
keine unpassende Todesbotin. Die Jahre hatten ihren Leib gekrümmt,
alle ihre Glieder zitterten, denn sie war vom Schlage gerührt
worden, und ihr runzliges, entstelltes Antlitz glich mehr einer
grotesk-phantastischen Zeichnung als einem Werke aus den Händen der
Natur.

Ach! wie wenige alte Gesichter gibt es, die uns durch ihre Schönheit
erfreuen! Angst, Sorgen und Kümmernisse der Welt verwandeln das
menschliche Antlitz, wie sie die Herzen umwandeln, und erst wenn jene
schlummern und für immer vorüber sind, schwinden die unruhig bewegten
Wolken und verhüllen und verdunkeln den hellen Himmel nicht mehr. Es
ist sehr häufig bei den Gesichtern der Toten der Fall, daß sie selbst
in ihrer Erstarrung den längst vergessenen Ausdruck schlummernder
Kinder wieder annehmen und die Züge der Kinderjahre wieder bekommen,
so ruhig und friedlich wieder werden, daß diejenigen, die sie in ihrer
Kindheit gekannt, mit Ehrfurchtsschauern an ihren Särgen niederknien
und den Engel schon auf Erden schauen.

Die Alte humpelte ihrer keifenden Vorgesetzten voran, blieb endlich
keuchend stehen, um Atem zu schöpfen, und Mrs. Corney nahm ihr das
Licht aus der Hand und ging allein in das Zimmer der Sterbenden, in
welchem eine Lampe düster brannte. Am Krankenbette saß eine andere alte
Frau, und am Kamine stand der Lehrling des Apothekers und Doktors und
schnitt einen Zahnstocher aus einem Federkiel.

«Ein kalter Abend, Mrs. Corney», bemerkte der junge Herr, als die Dame
eintrat.

«Sehr kalt, in der Tat, Sir», erwiderte die Vorsteherin im höflichsten
Tone.

«Sie sollten bessere Kohlen von Ihren Lieferanten verlangen», sagte der
Apothekerlehrling; «diese hier taugen absolut nichts für ein so kaltes
Wetter.»

«Das ist Sache des Kollegiums, Sir», erwiderte die Dame.

Hier wurde das Gespräch durch das Stöhnen der Kranken unterbrochen.

«Oh,» sagte der junge Mann, indem er sein Gesicht dem Bette zugewandt,
«mit der ist's vorbei.»

«Wirklich?» fragte die Matrone.

«Ich würde mich darüber wundern, wenn sie noch eine Stunde lebte. Heda,
schläft sie, Alte?»

Die Wärterin nickte. Der Lehrling machte Gebrauch von seinem
Zahnstocher, während sich Mrs. Corney stumm an das Bett setzte, und
schlich nach einigen Minuten auf den Zehen hinaus. Gleich darauf
erschien auch die Wärterin wieder, die Mrs. Corney gerufen hatte,
winkte der anderen, und beide setzten sich an den Kamin und fingen
leise miteinander zu sprechen an.

«Hat sie noch mehr gesagt, Anny, wie ich fort war?»

«Kein Sterbenswörtchen.»

«Hat sie den gewärmten Wein getrunken, den ihr der Doktor verordnete?»

«Sie konnte keinen Tropfen hinunterbringen; ich trank ihn daher selbst
aus, und er hat mir sehr gut geschmeckt.»

«Ich weiß die Zeit noch sehr wohl, da sie's ebenso gemacht und
hinterher weidlich darüber gelacht hat.»

«Freilich; sie war 'ne lustige alte Seele, hat manch liebe Leiche
angekleidet und so hübsch ausstaffiert wie 'ne Wachspuppe. Ich hab' ihr
mehr als hundertmal dabei geholfen.»

Mrs. Corney hatte ungeduldig auf das Erwachen der Schlummernden
gewartet, stand auf, trat zu den beiden alten Megären und fragte
ärgerlich, wie lange sie denn eigentlich warten sollte.

«Nicht lange mehr, Mistreß. Wir brauchen nicht lange auf den Tod zu
warten. Geduld, Geduld! er wird uns allen bald genug kommen.»

«Halten Sie den Mund und sagen Sie mir, Martha, hat die Patientin
früher auch schon so gelegen?»

«Oft genug.»

«Wird's aber nicht wieder tun», fiel die andere Wärterin ein; «ich
meine, sie wird nur noch einmal wieder aufwachen, und wohl zu merken,
Mrs. Corney, nur auf eine kurze Zeit.»

«Ob sie auf eine lange oder kurze Zeit erwacht, sie wird mich nicht
hier finden. Ihr alle beide, belästigt mich nicht noch einmal um nichts
und wieder nichts, sonst geht's euch schlecht. Ich habe durchaus nicht
die Verpflichtung, alle alten Weiber im Hause sterben zu sehen, und was
noch mehr sagen will, ich mag's und will's nicht. Merkt euch das, ihr
unverschämten alten Schlumpen! Habt ihr mich noch einmal zur Närrin, so
nehmt euch in acht, das sag' ich euch.»

Sie ging hinaus, als ein Schrei der beiden Wärterinnen, die wieder an
das Bett getreten waren, sie zum Stillstehen brachte. Die Kranke hatte
sich kerzengerade emporgerichtet und streckte die Arme nach ihnen aus.
«Wer ist da?» rief sie mit hohler Stimme.

«Pst, pst! Legen Sie sich nieder», sagte eine der Wärterinnen.

«Ich lege mich lebendig nimmermehr, nimmermehr wieder nieder», rief die
Patientin. «Ich will mit ihr sprechen. Kommen Sie, Mrs. Corney, daß ich
Ihnen ins Ohr flüstern kann.»

Sie faßte die Vorsteherin beim Arme und drückte sie auf einen Stuhl,
der neben dem Bette stand, nieder und war im Begriff, zu sprechen,
als sie bemerkte, daß die beiden Wärterinnen so nahe wie möglich
herangetreten waren, um zu horchen, und sagte mit matter Stimme:
«Schicken Sie sie hinaus -- geschwind, o geschwind!»

Mrs. Corney befahl ihnen, hinauszugehen, und die Sterbende fuhr fort:
«Hören Sie mich nun an! In diesem selbigen Zimmer -- diesem selbigen
Bette lag einst eine hübsche, junge Frau. Sie ward mit blutenden Füßen,
staub- und schmutzbedeckt ins Haus gebracht, wurde von einem Knaben
entbunden und starb. Ich war ihre Wärterin. Ich will mich besinnen --
in welchem Jahre war es doch?»

«Auf das Jahr kommt's nicht an», unterbrach Mrs. Corney ungeduldig.
«Was haben Sie mir von ihr zu sagen?»

«Was ich von ihr zu sagen habe -- oh, ich weiß es wohl», murmelte
die Sterbende, richtete sich plötzlich mit gerötetem Gesicht und
vorspringenden Augen wieder empor und schrie fast: «Ich bestahl sie!
Sie war noch nicht kalt -- noch nicht kalt -- als ich's tat.»

«Sie bestahlen sie? -- Um Gottes willen, was nahmen Sie ihr?»

«Es -- das einzige, was sie hatte. Sie bedurfte Kleider, um sich vor
der Kälte zu schützen, und Speise, um nicht Hungers zu sterben, hatte
es aber trotzdem aufbewahrt, trug es im Busen; und es war von Gold, und
sie hätte sich damit vom Tode erretten können.»

«Gold! -- Weiter, weiter, Frau. Wer war die Mutter -- wann starb sie?»

«Sie gab mir den Auftrag, es aufzubewahren, und vertraute mir als der
einzigen Frau, die um sie war. Ich stahl es ihr schon in Gedanken,
als sie's mir zeigte; und vielleicht bin ich auch am Tode des Kindes
schuld! Man würde den Knaben besser behandelt haben, wenn man alles
gewußt hätte.»

«Alles gewußt! -- Sprechen Sie, sprechen Sie!»

«Der Knabe ward seiner Mutter so ähnlich, daß ich immer an sie denken
mußte, wenn ich ihn sah. Ach, die Ärmste! -- und sie war so jung -- und
so sanft und geduldig! Ich muß Ihnen aber noch mehr sagen -- noch viel
mehr; -- hab' ich's Ihnen noch nicht alles gesagt?»

«Nein, nein, nein -- nur schnell -- oder es wird zu spät werden!»

«Als die Mutter ihren Tod herannahen fühlte, flüsterte sie mir ins
Ohr, wenn das Kind am Leben bliebe, so würde der Tag erscheinen, wo es
sich beim Nennen des Namens seiner Mutter nicht beschimpft achten, und
Freunde finden --»

«Wie wurde das Kind getauft?»

«Oliver. Das Gold, das ich stahl -- war --»

«Was, ums Himmels willen, was war es?»

Frau Corney beugte sich in höchster Spannung über die Sterbende, die
noch ein paar unverständliche Worte murmelte und leblos auf das Kissen
zurücksank. --

«Mausetot!» bemerkte eine der Wärterinnen, als Frau Corney die Tür
wieder geöffnet hatte.

«Und hatte gar nichts zu erzählen», sagte Frau Corney und entfernte
sich, als wenn nur etwas ganz Gewöhnliches vorgegangen wäre.




25. Kapitel.

    Worin die Erzählung wieder zu Fagin und Konsorten zurückkehrt.


Während sich die erzählten Ereignisse im Armenhause zutrugen, kauerte
Fagin brütend an einem matten, rauchigen Feuer in seiner alten Höhle
-- derselben, aus welcher Oliver von Nancy entfernt worden war. Er
hielt einen Blasebalg auf seinen Knien, mit dem er sich augenscheinlich
bemühte, das Feuer zu hellerer Flamme anzufachen. Aber er war in
tiefe Gedanken versunken und blickte unverwandt, die Ellbogen auf den
Blasebalg gestützt und das Kinn auf seinen Daumen ruhen lassend, auf
das rostige Gitter.

An einem Tische hinter ihm saßen der gepfefferte Baldowerer, Charley
Bates und Tom Chitling bei einer Partie Whist. Der Baldowerer spielte
mit dem Strohmanne und gewann fortwährend, die Karten mochten fallen,
wie sie wollten. Chitling zahlte, sprach seine Verwunderung über
Dawkins' stets glückliches Spiel aus und erklärte, daß nicht gegen
ihn «anzukommen» sei. Charley Bates lachte ausgelassen, und Fagin
blickte auf und bemerkte, Tom müsse sehr früh aufstehen, um gegen den
Baldowerer zu gewinnen.

«Ja, du mußt früh aufstehen, wenn du das willst, Tom,» fiel Charley
ein, «und obendrein die Stiefel über Nacht anbehalten und 'ne doppelte
Brille aufsetzen.»

Dawkins hörte die ihm gezollten Lobsprüche mit philosophischem
Gleichmute an, und zeichnete sinnig den Grundriß vom Newgategefängnis
mit Kreide auf den Tisch.

«Du bist grausam langweilig, Tommy», sagte der Baldowerer nach einer
Pause von mehreren Minuten. «Woran sollte er wohl denken, Fagin?»

«Wie kann ich's wissen?» antwortete der Jude. «Vielleicht an seinen
Verlust oder seinen angenehmen Aufenthalt auf dem Lande, woher er
gekommen ist erst soeben. Ha, ha, ha! Ist's das?»

«Falsch geraten», fuhr der Baldowerer fort. «Was meinst du, Charley?»

«Nun, ich meine,» erwiderte Master Bates grinsend, «daß er zuckersüß
gegen Betsy war. Schau, wie rot er wird! 's ist zum Totlachen -- Tommy
verliebt! O Fagin, Fagin, welch ein Hauptspaß!»

«Laß ihn zufrieden», sagte der Jude, Dawkins einen Wink gebend und
Bates einen mißbilligenden Stoß mit dem Blasebalg versetzend. «Betsy
ist 'ne schmucke Dirne. Mach dich immerhin an sie, Tom; mach dich
immerhin an sie 'ran!»

«Fagin,» nahm Chitling zornig das Wort, «das geht hier niemand was an.»

«O nein», erwiderte der Jude. «Laß Charley doch schwatzen und lachen;
er läßt's einmal nicht. Betsy ist 'ne artige Dirne. Tu, was sie dir
sagt, Tom, und du wirst machen dein Glück.»

«Ich tue, was sie mir sagt,» fuhr Tom fort, «und wäre nicht in die
Tretmühle gesteckt worden, hätt' ich ihren Rat nicht befolgt. Ihr habt
aber am Ende 'nen guten Rebbes dabei gemacht -- nicht wahr, Fagin?
Und was wollen sechs Wochen sagen? Es kommt doch einmal, früher oder
später, und im Winter ist's just am besten, wenn einem nicht daran
gelegen ist, so oft auszugehen -- he, Fagin?»

«Sehr richtig, mein Lieber», versetzte der Jude.

«Es wird dir gewiß gleich viel ausmachen, Tom, noch einmal in die Mühle
zu kommen,» fiel der Baldowerer, Fagin und Bates zublinzelnd, ein,
«wenn nur alles mit Betsy in Richtigkeit wäre.»

«Ja, das würd's -- seht!» erwiderte Tom noch erzürnter, «und ich möchte
doch wissen, wer mir's nachtäte, Fagin?»

«Das fällt ein keiner Seele», antwortete Fagin. «Ich weiß keinen außer
dir, der's würde tun.»

«Ich hätte ganz davonkommen können, hätt' ich mosern wollen -- he,
Fagin?» fuhr der halb blödsinnige Bursche, immer zorniger werdend,
fort. «Ich hätte nur ein einziges Wort zu sagen brauchen, nicht wahr,
Fagin? Ich schwatzte aber nicht -- und was ist denn nun dabei zu
lachen?»

Fagin eilte, ihm zu versichern, daß niemand lache, nicht einmal Charley
Bates, der jedoch, als er den Mund öffnete, um auch seinerseits zu
erklären, daß alle ohne Ausnahme äußerst ernsthaft gestimmt wären, in
ein unbezähmbares Gelächter ausbrach. Tom Chitling sprang wütend auf,
um dem Frechen einen Schlag zu versetzen, allein Charley bückte sich
gewandt, und der Schlag traf den munteren alten Herrn dermaßen vor
die Brust, daß derselbe gegen die Wand taumelte, und daß ihm der Atem
verging.

«Still! ich hab' den Bimbam g'hört», rief der Baldowerer in diesem
Augenblick, nahm das Licht vom Tische, schlich leise die Treppe hinauf,
kehrte nach einer halben Minute zurück und flüsterte Fagin etwas in das
Ohr.

«Wie?» rief der Jude. «Allein?»

Der Baldowerer nickte und gab Charley Bates einen freundschaftlichen
Wink, er täte besser daran, seine Heiterkeit etwas zu zügeln. Dann
blickte er wieder den Juden an und erwartete dessen Anweisungen.

Der alte Mann biß sich auf seine gelben Finger und sann einige
Augenblicke nach. Sein Gesicht arbeitete währenddessen heftig, als sei
er erschrocken und fürchte, das Schlimmste zu erfahren. Endlich erhob
er den Kopf und fragte: «Wo ist er?»

Der Baldowerer deutete nach oben und machte Miene, das Zimmer zu
verlassen.

«Ja,» sagte der Jude als Antwort auf diese stumme Frage, «bring ihn
herunter. Pst, still, Charley und Tom, still, still!»

Die Angeredeten gehorchten sofort. Sie gaben keinen Laut von sich, als
der Baldowerer, das Licht in der Hand, die Treppe herabkam und ihm
dicht auf den Fersen ein Mann folgte, der, nachdem er sich hastig im
Zimmer umgeblickt hatte, ein großes Tuch abwarf, das bisher den unteren
Teil seines Gesichts verdeckte, so daß die hageren, ungewaschenen und
unrasierten Züge des blonden Toby zum Vorschein kamen. Er begrüßte
Fagin, der ihn ängstlich fragend ansah, und erklärte sogleich, von
Geschäften nicht eher reden zu können, als bis er gegessen und
getrunken hätte. Der Jude befahl Dawkins, aufzutragen, was vorhanden
wäre; es geschah, und Toby machte sich begierig darüber her, ohne die
mindeste Neigung zu zeigen, das Gespräch zu beginnen und der Ungeduld
und Herzensangst des Juden ein Ende zu machen, der auf und ab laufend
mit seinen Blicken jeden Bissen zählte und verwünschte, den Toby zum
Munde führte. Toby lächelte, während er speiste, selbstgefällig und
schmunzelnd wie immer, und der Jude hätte vor Ingrimm vergehen mögen.
Endlich hub er an: «Vor allen Dingen, Fagin --»

«Ja, ja doch -- vor allen Dingen --»

«Vor allen Dingen, Fagin, wie steht's mit Bill?»

«Wie -- mit Bill!» kreischte der Jude, vom Stuhle aufspringend, denn er
hatte sich hörbegierig dicht neben Toby gesetzt.

«Zum Geier -- Ihr wollt doch nicht sagen --» fuhr Crackit erblassend
fort.

«Was soll ich nicht wollen sagen?» schrie der Jude, wütend mit den
Füßen stampfend. «Wo sind sie? -- Sikes und der Knabe -- wo sind sie?
-- wo sind sie geblieben? -- wo sind sie versteckt? -- warum sind sie
nicht hier?»

«Der Einbruch mißglückte», erwiderte Toby mit unsicherer Stimme.

«Ich weiß es», sagte der Jude, ein Zeitungsblatt aus der Tasche nehmend
und es Toby vorhaltend. «Was weiter?»

«Es wurde geschossen und der Knabe getroffen. Wir machten uns mit ihm
davon -- rannten und setzten über Hecken und Gräben, als wenn der
Teufel selbst hinter uns wäre. Wir wurden verfolgt -- Gott verdamm'
mich, die ganze Umgegend war lebendig, und wir hatten die Hunde auf den
Fersen.»

«Aber der Knabe, der Knabe!» keuchte Fagin.

«Bill trug ihn auf dem Rücken; wir hielten an mit Laufen, um ihn
zwischen uns zu nehmen; er ließ den Kopf hängen und war steif und
kalt. Sie waren dicht hinter uns, und da galt's, jeder sich selbst der
Nächste, wenn er nicht der erste am Galgen sein wollte. Wir rissen
aus, der eine hier, der andere da hin, und ließen den Burschen in 'nem
Graben liegen -- ob tot oder lebendig, ich kann's nicht sagen. Das ist
alles, was ich von ihm weiß.»

Der Jude stieß einen gellenden Schrei aus, fuhr mit den Händen in das
Haar und stürzte aus dem Zimmer und zum Hause hinaus.




26. Kapitel.

    In welchem eine geheimnisvolle Person auftritt und viel von der
    Erzählung Untrennbares geschieht.


Der alte Mann hatte die Straßenecke erreicht, bevor er anfing, sich von
dem Schrecken wieder zu erholen, den ihm Tobys Mitteilungen eingejagt
hatten. Er eilte soviel wie möglich durch Nebenstraßen und Gassen,
fast sinnlos immer vorwärts, so daß er beinahe von einem Mietswagen
überfahren worden wäre, und langte endlich auf Snow-Hill an, wo er
seine Schritte noch beschleunigte, bis er in eine lange und enge Gasse
eingebogen war. Jetzt schien er sich auf seinem Terrain zu fühlen und
freier zu atmen, denn er lief nicht mehr, sondern verfiel in seinen
gewöhnlichen, halb trippelnden, halb schlürfenden Gang.

Nicht weit von der Stelle, wo Snow-Hill und Holborn-Hill
zusammenstoßen, öffnet sich rechter Hand, wenn man aus der City
kommt, eine nach Saffron-Hill führende enge und erbärmliche Straße
-- Field-Lane -- mit zahllosen schmutzigen Läden, in welchen die
Taschentücher feilgeboten werden, welche die Ladenbesitzer von den
Taschendieben erhandelt haben. Die Straße hat ihren eigenen Barbier,
ihr Kaffeehaus, ihre Bierstube und ihre Garküche. Sie bildet eine
eigene Handelskolonie, ist der Stapelplatz für tausenderlei Artikel,
die Industriefrüchte der kleineren Diebe, und wird am frühen Morgen
und in der Abenddämmerung von schweigsamen Handelsleuten besucht,
die in finsteren Hinterzimmern ihre Geschäfte abmachen und auf so
absonderliche Art gehen, wie sie kommen.

In Field-Lane lenkte der Jude ein. Er war den Bewohnern sehr wohl
bekannt, von denen einer nach dem andern dem Vorübergehenden
vertraulich zunickte. Er erwiderte ihre Begrüßungen auf dieselbe Weise,
hielt sich indes nirgends auf, bis er den Ausgang der Straße erreicht
hatte, wo er einen Handelsmann von sehr kleiner Statur anredete, der in
seinem Laden saß und behaglich seine Pfeife rauchte. Er fragte ihn, wie
er sich befände.

«Vortrefflich! Aber in aller Welt, Mr. Fagin, wie, bekommt man Euch
einmal wieder zu sehen?» erwiderte das Männchen.

«Die Nachbarschaft hier war zu heiß ein wenig, Lively!» sagte Fagin,
die Augenbrauen emporziehend und die Hände über der Brust kreuzend.

«Hm! ich habe wohl schon ein paarmal darüber klagen hören; sie kühlt
sich indes bald wieder ab -- findet Ihr das nicht auch?»

Fagin nickte, wies nach Saffron-Hill und fragte, ob dort zu Abend
jemand wäre.

«In den Krüppeln?» fragte der kleine Handelsmann.

Der Jude bejahte.

«Wartet mal», fuhr der Handelsmann nachsinnend fort. «Ja, es ist ein
halbes Dutzend hineingegangen, soviel ich gesehen habe. Ich glaube aber
nicht, daß Euer Freund dort ist.»

«Ist Sikes nicht da?» fragte Fagin mit der Miene getäuschter Erwartung.

«Nein», erwiderte der Kleine, mit einem unsagbar schlauen Ausdruck den
Kopf schüttelnd. «Habt Ihr nichts zu handeln heute?»

«Heute nicht», erwiderte der Jude im Fortgehen.

«Geht Ihr in die Krüppel, Fagin?» rief ihm der kleine Handelsmann nach.
«Ich will mitgehen und 'nen Tropfen mit Euch trinken.»

Fagin winkte ihm mit der Hand, ihm bedeutend, daß er allein zu bleiben
wünsche, und die Krüppel wurden somit für dieses Mal der Ehre des
Besuchs Mr. Livelys beraubt, zumal der kleine Mann nicht leicht von
seinem Geschäft abkommen konnte. Während er sich erhoben hatte, war
der Jude verschwunden, und nachdem Mr. Lively sich vergebens auf die
Zehen gestellt hatte, um ihn nochmals zu Gesicht zu bekommen, mußte er
sich notgedrungen wieder auf seinen Stuhl setzen und nahm nach einem
bedenklichen und mißtrauischen Kopfschütteln seine Pfeife wieder zur
Hand.

Die Krüppel waren das Gasthaus, in welchem Sikes und sein Hund bereits
figuriert haben. Fagin gab einem Manne am Schenktische nur ein stummes
Zeichen und ging geradeswegs die Treppe hinauf, öffnete eine Tür, trat
sacht hinein und blickte ängstlich suchend und die Augen mit der Hand
beschattend, umher.

Das Zimmer war durch zwei Gasflammen erleuchtet, man hatte aber die
Fensterläden verschlossen und die Vorhänge dicht zugezogen. Die
Decke war geschwärzt, damit ihre Farbe unter dem Qualm der Lampen
nicht litte, und der ganze Raum dergestalt mit Tabaksrauch angefüllt,
daß Fagin anfangs kaum einen Gegenstand zu unterscheiden vermochte.
Allmählich erkannte er jedoch die zahlreiche Gesellschaft, deren
Anwesenheit ihm zuerst nur durch verworrenen Lärm kund geworden war.
Oben an der Tafel saß mit einem Präsidentenhammer der Wirt, ein
plumper, vierschrötiger Mann, der, als ein munteres Lied gesungen
wurde, sich gänzlich der allgemeinen Heiterkeit hinzugeben schien,
die Augen und Ohren aber -- und zwar sehr scharfe Augen und Ohren
-- offen und überall hatte. Ihm gegenüber an einem verstimmten
Fortepiano saß ein Musiker mit bläulicher Nase und Zahnschmerzen
halber verbundener Wange. Die Sänger ließen sich ihre Gläser noch
weit besser als die ihnen gespendeten Lobsprüche behagen, und die
Gesichter ihrer Bewunderer drückten fast jedes Laster in jeglicher
Abstufung aus und waren unwiderstehlich anziehend, weil grenzenlos
abstoßend. Man sah überall die mannigfachsten und wahrhaftesten
Bilder der Verschmitztheit, Brutalität und Trunkenheit, und die --
sämtlich noch mehr oder minder jugendlichen -- Frauenzimmer trugen die
abschreckendsten Spuren der Ausschweifung an sich, während in ihrem
wüsten Aussehen keine Spur edler Weiblichkeit mehr zu entdecken war, so
daß sie die schwärzeste und betrübendste Schattenpartie des Gemäldes
bildeten.

Fagin ließ sich jedoch durch Gedanken solcher Art nicht von fern
beunruhigen. Seine Blicke schweiften gespannt von einem Gesicht zum
andern, schienen aber vergebens zu suchen. Er winkte endlich unbemerkt
dem vorsitzenden Wirte, und schlich so sacht wieder hinaus, wie er
hineingeschlichen war.

«Was wünscht Ihr von mir, Mr. Fagin?» fragte der Wirt leise, sobald er
beim Juden draußen an der Treppe stand. «Wollt Ihr Euch nicht zu uns
setzen? Die ganze Gesellschaft würde sich sehr freuen.»

Der Jude schüttelte ungeduldig den Kopf und flüsterte: «Ist er hier?»

«Nein.»

«Keine Nachricht von Barney?»

«Nein. Er wird sich auch nicht rühren, bis alles sicher ist. Verlaßt
Euch drauf, sie sind ihm auf der Spur, und wenn er sich blicken ließe,
würde er die ganze Geschichte verraten. 's ist alles ganz richtig mit
ihm: ich hätte sonst von ihm gehört. Laßt ihn nur zufrieden; ich stehe
dafür, daß er sich mit großer Klugheit benimmt.»

«Wird er nicht kommen heut' abend?»

«Meint Ihr Monks?» lautete des Wirtes zögernde Gegenfrage.

«Pst! Ja doch!»

«Ich hab' ihn schon erwartet, und wenn Ihr nur zehn Minuten verweilen
wollt --»

«Nein, nein», unterbrach ihn der Jude hastig, als ob es ihn beruhigt
hätte, zu hören, daß der Mann, nach welchem er gefragt, nicht anwesend
sei, so begierig er, wie es schien, gewesen war, ihn zu sehen. «Sagt
ihm, daß ich ihn gesucht hätte hier, und daß er noch heute abend müßte
kommen zu mir -- doch nein, sagt morgen. Da er einmal nicht hier ist,
wird's auch morgen noch sein Zeit genug.»

«Gut! Habt Ihr noch ein Anliegen?»

«Nein, gute Nacht!» erwiderte Fagin im Hinuntergehen.

«Holla!» rief ihm der Wirt flüsternd nach, «was dies für 'ne
Gelegenheit zu 'nem Geschäftchen sein würde! Ich hab' da den Phil
Barker drinnen so sternig[AK], daß ihn ein Kind brennen[AL] könnte.»

  [AK] betrunken.

  [AL] betrügen.

«Ah so! 's ist aber noch nicht für Phil Barker die Zeit», rief der Jude
ebenso leise zurück. «Phil hat noch zu tun etwas, bis wir können ihn
entbehren. Geht also wieder zu Eurer Gesellschaft, mein Lieber, und
sagt den Leuten, daß sie lustig möchten leben -- solange sie noch am
Leben sind. Ha, ha, ha!»

Der Wirt stimmte in das heisere Lachen des alten Mannes ein und kehrte
zu seinen Gästen zurück. Sobald der Jude allein war, wurden auch seine
Mienen wieder nachdenklich und besorgt. Nach einem kurzen Besinnen rief
er einen Mietskutscher an, befahl ihm, nach Bethnal Green zu fahren,
stieg einige tausend Schritte vor Sikes' Wohnung wieder aus und eilte
zu Fuß weiter.

«Jetzt wird sich's schon zeigen, mein Mädchen,» murmelte er vor sich
hin, als er an die Haustür klopfte; «führst du was Geheimes im Schilde,
so will ich's bald haben heraus, so listig du auch bist.»

Er schlich leise hinauf und trat, ohne anzuklopfen, in Nancys Zimmer.
Sie war allein und lag mit dem Kopfe, um den das Haar unordentlich
herumhing, auf dem Tische. «Sie hat getrunken», dachte er gleichgültig,
«oder ist vielleicht bloß unwirsch.»

Der alte Mann drückte die Tür wieder zu, während er diese Betrachtung
anstellte, und das dadurch hervorgebrachte Geräusch weckte sie aus
ihrem Schlummer oder Hinbrüten; sie begegnete ruhig seinen forschenden
Blicken, fragte, was es Neues gäbe, und er erzählte ihr, was er von
Toby Crackit vernommen hatte. Sie hörte ihm zu, legte, ohne ein Wort
zu sprechen, den Kopf wieder auf den Tisch, stieß dann das Licht
ungeduldig von sich und scharrte mit den Füßen; dies war jedoch alles.

Der Jude blickte unruhig umher, als ob er sich überzeugen wollte, daß
Sikes nicht insgeheim zurückgekehrt wäre. Befriedigt, wie es schien,
durch sein Umherspähen, hustete er ein paarmal und machte ebensoviele
Versuche, ein Gespräch anzuknüpfen; allein das Mädchen beachtete ihn
nicht mehr, als wenn er eine Bildsäule gewesen wäre. Endlich nahm er
sich zusammen und sagte händereibend und im freundlichsten Tone: «Was
meinst du denn, liebes Kind, wo wohl sein mag Bill?»

Das Mädchen murmelte in kaum verständlichen Worten, sie könne es nicht
sagen, und es schien ihm, als ob sie leise schluchze.

«Und wo wohl mag sein der kleine Oliver?» fuhr er fort, die Augen
anstrengend, um etwas von ihrem Gesichte zu erspähen. «Das arme Kind --
denk nur, Nancy -- wie sie's haben lassen liegen in einem Graben!»

«Da ist ihm wohler als unter uns», sagte das Mädchen, plötzlich
aufblickend; «und wenn für Bill nichts Schlimmes daraus entsteht, so
will ich hoffen und wünschen, daß der Kleine tot im Graben liegt, und
daß seine jungen Gebeine darin verfaulen.»

Den Lippen des Juden entfloh ein Ausruf des Erstaunens.

«Ja, das hoff' und wünsch' ich», fuhr Nancy, seinen Blicken begegnend,
fort. «Ich freue mich, daß er mir aus den Augen, und zu wissen, daß
das Schlimmste vorüber ist. Ich *kann* ihn nicht um mich haben; ich
verabscheue mich selbst und euch alle, wenn ich ihn sehe.»

«Pah!» fiel der Jude verächtlich ein. «Du bist betrunken, Mädchen.»

«So -- betrunken!» höhnte Nancy. «Eure Schuld ist's freilich nicht,
wenn ich's nicht bin. Ich wäre niemals nüchtern, wenn's nach Eurem
Willen ginge, jetzt ausgenommen! -- Meine Laune scheint Euch nicht zu
behagen.»

«Nein, durchaus nicht!» sagte der Jude wütend.

«So ändert sie», fuhr das Mädchen mit Lachen fort.

«Sie ändern!» schrie der Jude, durch die unerwartete Hartnäckigkeit
des Mädchens und die Verdrießlichkeiten des Abends über alle Maßen
erbittert. «Ja, ich will sie ändern! Hör', was ich werde dir sagen,
du liederliches Weibsbild! Ich, der ich nur zu sprechen brauche sechs
Worte, und Sikes wird zugeschnürt die Kehle so gewiß, wie ich würde
ihn dämpfen, hätt' ich jetzt zwischen meinen Fingern seinen Stierhals.
Kommt er zurück, ohne mitzubringen den Knaben -- kommt er glücklich
davon und bringt mir nicht ihn, lebendig oder tot, Mädchen, so morde
deinen Bill selbst, wenn du willst, daß er entgehen soll dem Galgen,
und tu' es ja, sobald er den Fuß hier setzt hinein ins Zimmer; denn
merk', es wird sonst sein zu spät!»

«Was sagt Ihr da?» rief das Mädchen unwillkürlich aus.

«Was ich sage?» fuhr der Jude, vor Wut fast von Sinnen, fort. «Dies
sag' ich! Wenn das Kind ist wert viele hundert Pfund für mich, soll ich
verlieren, was mir zugewürfelt hat der Zufall, durch die Tollheiten
einer betrunkenen Bande, deren Leben in meiner Gewalt ist -- und indem
ich obenein gesellt bin mit 'nem eingefleischten Teufel, der nur
braucht zu wollen und hat die Macht, zu ... zu ...» --

Er keuchte atemlos, sprudelte vor Wut, bemühte sich vergebens, Worte
zu finden; plötzlich aber bezwang er seinen Zorn und nahm ein ganz
anderes Wesen an. Er sank zusammengekrümmt auf einen Stuhl nieder und
bebte vor Angst, geheimste Schurkereien selbst offenbart zu haben. Nach
einem kurzen Stillschweigen wagte er es, nach Nancy hinzublicken und
schien etwas ruhiger zu werden, als er sie wieder in derselben achtlos
gleichgültigen Stellung sah, in welcher er sie gefunden hatte.

«Nancy, liebes Kind,» krächzte er in seinem gewöhnlichen Tone, «hast du
gehört, was ich habe gesagt?»

«Laßt mich jetzt in Ruhe, Fagin», antwortete sie, den Kopf matt und
schläfrig emporrichtend. «Wenn es Bill diesmal nicht getan hat, so
wird er's ein andermal tun; er hat manch schönes Geschäft für Euch
ausgerichtet und wird Euch noch viele ausrichten, wenn er kann; kann
er's aber einmal nicht, so kann er's nicht. Und nun sprecht nicht mehr
davon.»

«Aber was anbelangt den Oliver, Kind?» sagte der Jude, indem er sich
unruhig die Hände rieb.

«Er muß das Schicksal der anderen teilen,» fiel Nancy hastig ein; «und
ich sag' es noch einmal, ich hoffe, daß er tot ist und vor Schaden und
vor Euch sicher ist -- das heißt, wenn Bill nichts Schlimmes begegnet;
und ist Toby gut davongekommen, so wird er's ohne Zweifel auch sein,
denn was der kann, kann Bill tausendmal.»

«Und was anbelangt das, was ich sagte, Kind?» sagte der Jude, sie
doppelt scharf in das Auge fassend.

«Ihr müßt's alles noch einmal wiederholen, wenn Ihr wollt, daß ich
etwas tun soll,» entgegnete Nancy, «und sagt mir es lieber morgen. Ihr
hattet mich auf 'nen Augenblick aufgestört, aber ich bin jetzt wieder
so müd' und dämlich wie vorher.»

Der Jude legte ihr noch mehrere andere Fragen in derselben Absicht
vor, um zu erfahren, ob sie die ihm in einem unbewachten Augenblicke
entschlüpften Andeutungen beachtet und verstanden hätte; allein sie
antwortete und hielt seine forschenden Blicke so unbefangen aus, daß er
seinen ersten Gedanken, daß sie zuviel getrunken, vollkommen bestätigt
zu sehen glaubte. Und Miß Nancy war allerdings nicht frei von der unter
Fagins Zöglingen gewöhnlichen Schwäche, der Neigung zum übermäßigen
Genuß geistiger Getränke, in der sie in ihren zarteren Jahren eher
bestärkt wurden, als daß man sie davon zurückgehalten hätte. Ihr
wüstes Aussehen und der das Gemach anfüllende starke Genevergeruch
dienten zum bekräftigenden Beweise der Richtigkeit der Annahme des
Juden; und als sie endlich zu weinen und gleich darauf wieder zu
lachen anfing und wiederholt rief: «Heisa, wer wollte den Kopf hängen
lassen!» so zweifelte er, der in Sachen dieser Art seinerzeit große
eigene Erfahrungen gemacht hatte, nicht mehr und freute sich höchlich
der Gewißheit, daß ihre Trunkenheit in der Tat schon einen hohen Grad
erreicht hatte.

Er empfand infolge dieser Entdeckung eine große Erleichterung und
entfernte sich sehr zufrieden, seinen doppelten Zweck erreicht zu
haben, dem Mädchen zu hinterbringen, was ihm von Toby mitgeteilt
worden war, und sich mit eigenen Augen zu überzeugen, daß Sikes nicht
zurückgekehrt wäre. Es war eine Stunde vor Mitternacht und bitterlich
kalt; er säumte daher nicht, seine Wohnung baldmöglichst zu erreichen.
Als er an der Ecke der Straße, in welcher sie lag, angelangt war und
schon in der Tasche nach dem Hausschlüssel suchte, trat plötzlich und
unhörbar ein Mann hinter ihn und flüsterte seinen Namen dicht an
seinem Ohre. Er wendete sich rasch um und sagte: «Ist das --»

«Ja, ich bin's», unterbrach ihn der Mann barsch. «Hab' hier seit zwei
Stunden aufgepaßt. Wo zum Teufel seid Ihr gewesen?»

«Beschäftigt mit Euren Angelegenheiten, mein Lieber», erwiderte der
Jude, ihn unruhig anblickend und einen langsameren Schritt annehmend.
«Den ganzen Abend beschäftigt mit Euren Angelegenheiten.»

«Ei, natürlich», sagte der andere höhnisch. «Was habt Ihr denn
ausgerichtet?»

«Nicht viel Gutes», antwortete Fagin.

«Ich will hoffen, nichts Schlimmes», fiel der Vermummte, stillstehend
und den Juden wild ansehend, ein.

Fagin schüttelte den Kopf und stand im Begriff, ihm eine Antwort zu
geben, als ihn der Vermummte unterbrach und sagte, er wolle lieber
drinnen im Hause anhören, was er würde hören müssen, denn er wäre halb
erfroren. Der Jude sah ihn mit einer Miene an, die offenbar genug
verkündete, daß er des Besuches zu einer so späten Stunde gar gern
überhoben wäre, und murmelte, daß er kein Feuer habe, und Ähnliches;
allein der unwillkommene Gast wiederholte seine Erklärung, mit ihm
gehen zu wollen, mit großer Bestimmtheit, und Fagin schloß die Haustür
auf und sagte ihm, er möge sie leise wieder verschließen, während er
selbst Licht holen wolle.

«'s ist hier so finster wie im Grabe», bemerkte der Besucher, ein paar
Schritte vorwärts tappend. «Macht geschwind, ich kann solche Dunkelheit
nicht leiden.»

«Verschließt die Tür», flüsterte Fagin unten auf dem Hausflur, und
während er sprach, wurde die Türe mit donnerndem Schalle zugeworfen.

«Das hab' ich nicht getan», sagte Fagins Peiniger, sich vorwärts
fühlend. «Der Wind schlug sie zu, oder sie schloß sich von selber.
Macht geschwind, daß Ihr Licht bekommt, oder ich stoße mir in diesem
verwünschten Loche den Kopf noch ein.»

Fagin schlich in die Küche hinunter und kehrte bald darauf mit einem
angezündeten Lichte und der Kunde zurück, daß Toby Crackit unten im
Hinter- und die Knaben im Vorderzimmer schliefen. Er winkte seinem
ungebetenen Gaste und führte ihn die Treppe hinauf in ein Zimmer des
oberen Stockwerks.

«Wir können sagen hier die paar Worte, die wir haben zu sagen,» begann
er, als sie eingetreten waren, «und ich will das Licht setzen draußen
an die Treppe, denn in den Fensterläden sind Löcher, und wir lassen
niemals sehen die Nachbarn, daß wir Licht haben.»

Er stellte den Leuchter der Tür des Zimmers gegenüber, in welchem sich
nur ein gebrechlicher Sessel und hinter der Tür ein altes Sofa ohne
Überzeug befand, auf das sich der müde Fremde warf. Der Jude setzte
sich vor ihn in den Sessel. Da die Tür halb offen stand, so war es im
Zimmer nicht ganz finster, und das draußen stehende Licht warf einen
schwachen Schein auf die Wand gegenüber.

Sie flüsterten einige Zeit so leise miteinander, daß ein Horcher
von ihrer Unterredung nur etwa so viel hätte verstehen können, um
daraus zu entnehmen, daß sich Fagin gegen Beschuldigungen des Fremden
verteidigte, und daß sich dieser in einer sehr gereizten Stimmung
befand. Sie mochten etwa eine Viertelstunde geflüstert haben, als Monks
-- denn so hatte der Jude seinen Besucher mehrere Male genannt -- etwas
lauter sagte: «Ich wiederhol's Euch, es war schlecht ausgedacht. Warum
habt Ihr ihn nicht hier behalten bei den anderen und ohne weiteres 'nen
jämmerlichen Taschendieb aus ihm gemacht?»

«Hör' einer an!» rief der Jude achselzuckend aus.

«Wollt Ihr damit sagen, daß Ihr's nicht gekonnt hättet, wenn Ihr
gewollt?» fragte Monks unwillig. «Habt Ihr's nicht bei hundert anderen
Knaben verstanden? Hättet Ihr höchstens zehn bis zwölf Monate Geduld
gehabt, so wär's Euch doch ein leichtes gewesen, zu machen, daß er
verurteilt und vielleicht auf Lebenszeit deportiert wurde.»

«Wem würde dabei gewesen sein gedient, mein Lieber?» fragte der Jude im
demütigsten Tone.

«Mir!»

«Aber mir nicht», fuhr Fagin fast noch unterwürfiger fort. «Wenn zwei
Leute sind beteiligt bei einem Geschäft, so ist's doch nur billig, daß
berücksichtigt wird der Vorteil beider.»

«Was weiter?»

«Ich sah, daß es nicht leicht war, ihn zu erziehen zum Geschäft; er
hatte nicht denselben Charakter wie andere Knaben.»

«Hol' ihn der Satan, nein! denn er wäre sonst schon längst ein
Spitzbube gewesen.»

«Ich hatte kein Mittel in Händen, ihn zu machen schlimmer», fuhr der
Jude, angstvoll Monks Mienen beobachtend, fort; «er hatte in nichts die
Hand drin; ich konnt' ihm mit gar nichts einjagen Furcht und Schrecken,
und wir arbeiten immer vergeblich, wenn das nicht angeht. Was konnt'
ich tun? Ihn ausschicken mit dem Baldowerer und Charley? Es geschah,
und wir hatten genug an dem einen Male, mein Bester; ich mußte zittern
für uns alle.»

«*Das* war meine Schuld nicht», bemerkte der finstere Monks.

«Freilich; nein, o nein, mein Lieber, und ich mache Euch auch keinen
Vorwurf deshalb; denn wär's nicht geschehen, so wären Eure Blicke
vielleicht nicht gefallen auf den Knaben, und wir hätten vielleicht
niemals gemacht die Entdeckung, daß er es war, den Ihr suchtet. Nun
gut; ich bracht' ihn wieder in meine Gewalt durch die Nancy, und jetzt
fängt sie an und wirft sich auf zu seiner Freundin.»

«Schnürt ihr die Kehle zu!» sagte Monks ungeduldig.

«Geht jetzt eben nicht an, mein Lieber,» versetzte Fagin lächelnd; «und
außerdem machen wir in dergleichen keine Geschäfte, sonst wär mir's
schon lieb, wenn es geschähe über kurz oder lang. Monks, ich kenne
diese Dirne, sobald anfängt der Knabe verhärtet zu werden, wird sie
sich nicht kümmern um ihn mehr, als um 'nen Holzblock. Ihr wollt, daß
er werden soll ein Dieb; ist er noch am Leben, so kann ich ihn jetzt
dazu machen; und wenn -- wenn -- 's ist freilich nicht wahrscheinlich
-- aber wenn sich das Schlimmste hat ereignet, und er ist tot --»

«Wenn er's ist, so ist's meine Schuld nicht!» unterbrach ihn Monks mit
bestürzter Miene und mit bebender Hand den Juden beim Arme fassend.
«Merkt wohl, Fagin! ich habe keine Hand dabei im Spiel gehabt. Ich
hab's Euch von Anfang an gesagt, alles -- nur nicht, daß er sterben
sollte. Ich mag kein Blut vergießen -- es kommt stets heraus und
peinigt einen außerdem! Ist er totgeschossen, so kann ich nichts dafür;
hört Ihr, Fagin? -- Was -- ist der Teufel in dieser verwünschten
Spelunke los? -- was war das?»

«Was -- in aller Welt?» schrie der Jude, Monks mit beiden Armen
umfassend, als derselbe plötzlich im höchsten Schrecken emporsprang.
«Was -- wo?»

«Dort!» erwiderte der bebende Monks, nach der Wand gegenüber
hinzeigend. «Der Schatten -- ich sah den Schatten eines Frauenzimmers
in 'nem Mantel und Hut, wie 'nen Hauch an dem Täfelwerk dahingleiten.»

Der Jude ließ ihn los, und beide stürzten aus dem Zimmer hinaus.
Das vom Zugwinde flackernde Licht, das an der Stelle stand, wo es
Fagin hingestellt hatte, zeigte ihnen nur die leere Treppe und
ihre erbleichten Gesichter. Sie horchten mit der gespanntesten
Aufmerksamkeit, allein die tiefste Stille herrschte im ganzen Hause.

«'s ist nichts gewesen als Eure Einbildung», sagte der Jude, das Licht
aufhebend und zu Monks sich wendend.

«Ich will darauf schwören, daß ich's wirklich sah», versetzte Monks,
fortwährend heftig zitternd. «Es beugte sich vor, als ich's erblickte,
und verschwand, als ich zu Euch davon zu sprechen anfing.»

Der Jude warf ihm einen verächtlichen Blick zu, forderte ihn auf, ihm
zu folgen, wenn es ihm beliebe, und ging voran die Treppe hinauf. Sie
schauten in alle Gemächer hinein, begaben sich wieder hinunter auf den
Hausflur, in die Keller, durchsuchten jeden Winkel, allein vergebens.
Es war im ganzen Hause öde und still wie der Tod.

«Was meint Ihr nun, mein Guter?» sagte der Jude, als sie wieder auf dem
Hausflur standen. «'s ist im Hause kein lebendiges Wesen außer uns und
Toby Crackit und den Knaben, und die sind wohl verwahrt. Schaut!»

Er nahm zwei Schlüssel aus der Tasche und fügte hinzu, daß er, als
er zuerst hinuntergegangen, Toby, Dawkins und Charley eingeschlossen
habe, um jede Störung des Gesprächs unmöglich zu machen. Monks
wurde wankend in seinem Glauben und erklärte endlich, daß ihm seine
erhitzte Einbildungskraft einen Streich gespielt haben müsse, wollte
die Unterredung jedoch für diesmal nicht fortsetzen, erinnerte sich
plötzlich, daß ein Uhr vorüber sei, und das liebenswürdige Freundespaar
trennte sich.




27. Kapitel.

    In dem die Unhöflichkeit eines früheren Kapitels bestmöglich wieder
    gutgemacht wird.


Da es der geringen Person eines Schriftstellers schlecht anstehen
würde, einen so wichtigen Mann wie einen Kirchspieldiener mit den
Rockschößen unter dem Arme am Feuer stehen zu lassen, bis es dem Autor
eben beliebte, ihn zu erlösen; und da ihm seine Stellung oder seine
Galanterie noch weniger erlaubt, auf ähnliche Weise eine Dame zu
vernachlässigen, auf welche besagter Kirchspieldiener ein wohlgeneigtes
und zärtliches Auge geworfen und in deren Ohren er süße Worte
geflüstert, welche, aus dem Munde eines solchen Mannes kommend, in den
Herzenssaiten jeglicher Jungfrau oder Matrone Anklang finden mußten: so
eilt der gewissenhafte Erzähler dieser Geschichte, der die gebührende
Ehrfurcht vor denjenigen hegt, welche mit hoher und wichtiger
Autorität bekleidet sind, ihnen jene Achtung zu zollen, welche ihre
Stellung erfordert, und ihnen die ganze pflichtmäßige, rücksichtsvolle
Behandlung angedeihen zu lassen, zu welcher ihr hoher Rang und folglich
ihre großen Tugenden sie auf das vollkommenste berechtigen. Es war
seine Absicht, zu diesem Zwecke hier eine Abhandlung einzufügen, in
welcher das göttliche Recht der Kirchspieldiener erörtert und der Satz,
daß ein Kirchspieldiener kein Unrecht tun könne, ins Licht gestellt
werden sollte, -- eine Abhandlung, die für den verständigen und
wohlgesinnten Leser sowohl angenehm wie nützlich hätte werden müssen;
allein der Mangel an Zeit und Raum nötigt ihn unglücklicherweise,
sie für jetzt noch zurückzustellen. Sobald es indes an Zeit und Raum
nicht mehr gebricht, wird er zeigen, daß ein Kirchspieldiener in der
wahren und höchsten Potenz -- das will sagen ein solcher, der beim
Kirchspielarmenhause angestellt ist und in seiner amtlichen Eigenschaft
die Kirchspielkirche besucht -- nach den Rechten und kraft seines Amtes
alle Vortrefflichkeiten und mit einem Worte die besten Eigenschaften
der menschlichen Natur besitzt, und daß bloße Vereins- oder Kapellen-
oder Gerichtsdiener oder Pedelle auf jene Vortrefflichkeiten auch nur
die mindesten begründeten Ansprüche keineswegs machen können.

Mr. Bumble hatte wiederholt die Teelöffel gezählt, die Zuckerzange
gewogen, den Milchgießer geprüft und sämtliche Mobilien bis auf die
Pferdehaarkissen der Stühle einer genauen Besichtigung unterworfen,
ehe er daran dachte, daß es nachgerade wohl Zeit wäre, daß Mrs.
Corney zurückkehrte. Sie ließ jedoch noch immer nichts von sich weder
sehen noch hören, ein Gedanke pflegt einen anderen hervorzurufen,
und so dachte Mr. Bumble weiter, daß er sich zum Zeitvertreibe
nicht unschuldiger und gottseliger beschäftigen könne, als wenn er
seine Neugier durch einen flüchtigen Blick in Mrs. Corneys Kommode
befriedigte.

Nachdem er daher an der Tür gehorcht hatte, ob auch niemand in der Nähe
wäre, fing er seine Untersuchung bei der untersten Schublade an, und
die Kleider aus guten Stoffen, welche er fand, schienen ihm ausnehmend
zu gefallen. In der obersten entdeckte er eine verschlossene Büchse,
die er schüttelte, und das Geldgeklapper deuchte seinen Ohren gar
liebliche Musik. Nachdem er sich eine Zeitlang daran ergötzt hatte,
stellte er sich wie zuvor an den Kamin und sagte mit feierlich-ernster
Miene: «Ich tu's», schien durch ein schlaues, wohlgefälliges Lächeln
hinzufügen zu wollen, was er doch für ein rüstiger, lustiger und
pfiffiger alter Knabe sei, und betrachtete endlich mit vielem Vergnügen
und Interesse seine Waden im Profil.

Er war noch in sotane, befriedigende Wadenschau vertieft, als Mrs.
Corney hastig hereintrat, sich atemlos auf einen Stuhl am Kamin warf,
mit der einen Hand die Augen bedeckte, die andere auf das Herz legte
und nach Atem rang.

«Mrs. Corney,» sagte Bumble, sich über sie beugend, «was ist Ihnen,
Ma'am? Hat sich ein Unglück ereignet? Ich bitte, antworten Sie mir; ich
stehe hier auf -- auf --» Mr. Bumble konnte sich in seiner Bestürzung
nicht auf das Wort «Kohlen» besinnen, er sagte daher: «wie auf
Zuckerzangen».

«Oh, Mr. Bumble,» rief die Dame aus, «ich bin ganz wie zerschlagen!»

«Zerschlagen -- wie?» zürnte Bumble. «Wer hat sich unterfangen -- ah,
ich weiß es schon,» fügte er mit angeborener Würde und Feierlichkeit
hinzu, «abermals so ein Stück von den spitzbübischen, gottvergessenen
Armen!»

«'s ist schrecklich, nur daran zu denken!» sagte die Dame schaudernd.

«So denken Sie nicht daran, Ma'am», sagte Bumble.

«Ich kann's nicht lassen», entgegnete Frau Corney zimperlich.

«So stärken Sie sich durch einen Tropfen Wein», riet der
Kirchspieldiener in mitleidigem Tone.

«Nicht um die Welt!» erwiderte Mrs. Corney. «Es wäre mir ganz
unmöglich! Geistige Getränke -- nein -- nie -- Ach, ach! auf dem
obersten Simse rechter Hand; ach, ach!»

Die gute Frau hatte offenbar heftige Krämpfe und hatte schon die
Besinnung verloren, als sie nach dem Eckschranke hinwies. Bumble flog
auf denselben zu, fand eine grüne Flasche darin, nahm sie heraus,
füllte eine Tasse mit ihrem Inhalt und hielt sie der Dame an die Lippen.

«Mir ist wohler!» sagte Mrs. Corney, nachdem sie die Arznei halb
ausgetrunken hatte.

Bumble hob zum Zeichen seiner dankbaren Gefühle die Augen zur Decke
empor, senkte sie nieder auf den Rand der Tasse und hielt dieselbe
unter seine Nase.

«Pfefferminzwasser», sagte Mrs. Corney mit matter Stimme, aber dem
Kirchspieldiener zulächelnd. «Kosten Sie doch einmal -- es ist noch ein
wenig sonst was drin.»

Bumble kostete den heilkräftigen Trank, kostete noch einmal mit weiser,
prüfender Miene, und stellte die Tasse leer auf den Tisch.

«Es bekommt vortrefflich», bemerkte die Patientin.

Bumble erklärte, derselben Meinung zu sein, setzte sich neben Frau
Corney und fragte zärtlich: «Was ist Ihnen aber begegnet, Ma'am?»

«O nichts», erwiderte sie; «ich bin eine recht törichte, erregbare,
schwache Frau.»

«Schwach, Ma'am», sagte Bumble, ein wenig näher rückend. «Sind Sie
wirklich schwach, Mrs. Corney?»

«Wir sind alle schwache Geschöpfe», versetzte Mrs. Corney, einen
allgemeinen Satz aufstellend.

«Sehr wahr», stimmte Bumble ein.

Ein paar Minuten lang schwiegen beide, und nach Ablauf derselben hatte
Mr. Bumble den allgemeinen Satz praktisch dadurch erläutert, daß er
seinen linken Arm von Mrs. Corneys Stuhllehne entfernt und um ihr
Schürzenband gelegt, wo er nunmehr mit sanftem Drucke ruhte.

«Wir sind allesamt schwache Geschöpfe», wiederholte er.

Mrs. Corney seufzte.

«Seufzen Sie doch nicht, Ma'am!»

«Ach! Wenn ich's nur lassen könnte!» Sie seufzte abermals.

«Dies Zimmerchen ist sehr nett und behaglich, Ma'am. Es würde mit noch
so einem eine artige Wohnung ausmachen.»

«Es würde zu viel sein für eine einzelne Person», murmelte die Dame.

«Aber nicht für zwei, Ma'am», fiel Bumble schmachtend ein. «Was sagen
Sie, Mrs. Corney?»

Mrs. Corney senkte den Kopf bei diesen Worten Mr. Bumbles, und Mr.
Bumble senkte den seinigen gleichfalls, um ihr in das Gesicht schauen
zu können. Mrs. Corney blickte mit großer Züchtigkeit seitwärts, machte
ihre Hand los, um nach ihrem Taschentuche zu greifen, und ließ sie
unwillkürlich in die Hand Mr. Bumbles sinken.

«Gibt Ihnen die Direktion nicht freie Feuerung, Mrs. Corney?» fragte
der Kirchspieldiener, ihr zärtlich die Hand drückend.

«Und freies Licht», erwiderte Mrs. Corney, den Druck leise erwidernd.

«Feuerung, Licht und Wohnung frei», fuhr Bumble fort. «Oh, Mrs. Corney,
welch ein Engel Sie sind!»

Die Dame war gegen einen solchen Gefühlserguß nicht unempfindlich
genug, um noch länger widerstehen zu können, sondern sank in die Arme
Mr. Bumbles, welcher Gentleman ihr im Sturme seiner Gefühle einen
leidenschaftlichen Kuß auf die keusche Nase drückte.

«Oh, Sie Ausbund aller Kirchspielvollkommenheiten!» rief Mr. Bumble
ganz verzückt aus. «Sie wissen doch, meine Himmlische, daß Mr. Slout
heut' abend viel kränker geworden ist?»

«Ach ja», sagte Mrs. Corney verschämt.

«Der Doktor sagt, daß er keine acht Tage mehr leben könnte», fuhr
Bumble fort. «Sein Tod hat die Vakanz des Haushofmeisterpostens zur
Folge. Oh, Mrs. Corney, welche Aussichten eröffnen sich da! -- welche
Aussichten auf die allerseligste Herzens- und Haushaltsverschmelzung!»

Mrs. Corney schluchzte.

«O meine bezaubernde Mrs. Corney!» sprach Bumble weiter, «das kleine
Wörtchen -- nur das kleine, süße Wörtchen!»

«Ja -- a -- a!» hauchte Mrs. Corney.

«Und noch eins -- nur das eine noch -- wann soll es sein?»

Sie versuchte zweimal zu reden, doch vergebens. Endlich faßte sie
sich ein Herz, schlang die Arme um Bumbles Nacken und sagte, sobald
es ihm nur irgend gefiele, und er wäre ein gar zu lieber und ganz
unwiderstehlicher Mann.

Nachdem die Angelegenheit auf diese freundschaftliche und befriedigende
Weise geordnet war, wurde der Vertrag durch eine zweite Tasse
Pfefferminzwasser feierlich besiegelt, was bei der Erregtheit und
Beklemmung der Dame um so notwendiger war; und während die Tasse
geleert wurde, erzählte Mrs. Corney ihrem Zukünftigen von dem Tode der
alten Frau.

«Schön», bemerkte Bumble, sein Pfefferminzwasser schlürfend. «Ich will
auf meinem Rückwege nach Hause bei Sowerberry vorsprechen und die
erforderlichen Anordnungen treffen. Was war es denn aber, worüber Sie
so ganz außer sich zu sein schienen, meine Liebe?»

«Oh, es war nichts Besonderes, Bester», erwiderte die Dame ausweichend.

«Ei, es muß doch etwas Besonderes gewesen sein. Warum wollen Sie es
Ihrem Bumble nicht sagen?»

«Ein anderes Mal -- wenn wir erst verheiratet sind, mein Teuerster.»

«Wenn wir erst verheiratet sind! Es wird sich doch kein Armer eine
Unverschämtheit gegen Sie herausgenommen haben?»

«O nein, nein, durchaus nicht!» fiel die Dame hastig ein.

«Wenn ich das auch annehmen müßte,» fuhr der Kirchspieldiener fort,
«denken müßte, daß es ein Armer gewagt hätte, seine gemeinen Augen zu
dem liebenswürdigen Antlitze zu erheben --»

«Das hätte keiner gewagt -- nimmermehr --»

«Ich wollt's ihnen auch wohl raten!» zürnte Bumble, die Faust
schüttelnd. «Ich will den Menschen sehen, arm oder nicht arm, der
sich's unterfinge, und kann ihm nur so viel versichern, daß er's nicht
zum zweitenmal tun würde.»

Die Worte hätten vielleicht wie eine nicht eben große Schmeichelei
gegen die Reize der Dame geklungen, wenn sie nicht durch heftiges
Gebärdenspiel verschönt gewesen wären; da jedoch Bumble seine Drohung
mit vielen kriegerischen Gestikulationen begleitete, so erblickte
Mrs. Corney darin sehr gerührt nur einen Beweis seiner aufopfernden
Ergebenheit und versicherte ihm bewundernd und mit großer Wärme, daß er
wahrhaftig ein Täubchen wäre.

Mr. Bumble knöpfte den Rock bis unter das Kinn zu, setzte seinen
dreieckigen Hut auf, umarmte seine Taube zärtlich und lange und ging,
um abermals dem Sturme und der Kälte Trotz zu bieten, nachdem er zuvor
bloß noch fünf Minuten im Zimmer der männlichen Armen verweilt und
gegen dieselben ein wenig getobt hatte, um zu erproben, ob er der
Stelle des Haushofmeisters auch mit der gebührenden Autorität würde
vorstehen können. Nachdem er sich von seiner Befähigung überzeugt,
verließ er das Haus mit einem leichten, fröhlichen Herzen und
glänzenden Vorausahnungen seiner bevorstehenden Beförderung.

Mr. und Mrs. Sowerberry befanden sich in einer Abendgesellschaft, und
da Noah Claypole zu keiner Zeit geneigt war, sich einem größeren Maße
physischer Anstrengung zu unterziehen, als durch eine gemächliche
Betätigung der Funktionen des Essens und Trinkens erfordert wird, so
war der Laden noch nicht verschlossen, obgleich die Stunde längst
vorüber war, zu welcher es hätte geschehen sollen. Bumble klopfte
mehreremal mit seinem Rohre auf den Ladentisch; allein da niemand
erschien, und da er durch das Glasfenster des kleinen Zimmers hinter
dem Laden Licht schimmern sah, so trat er näher, um nachzusehen, was in
dem Zimmerchen vorginge, und war nicht wenig erstaunt, zu sehen, was er
sah.

Der Tisch war gedeckt, und auf ihm standen Brot und Butter, Teller
und Gläser, ein Krug mit Porter und eine Weinflasche. Noah Claypole
ruhte in nachlässigster Stellung in einem Sessel und hatte ein
mächtiges Butterbrot in der Hand. Dicht neben ihm stand Charlotte
und öffnete Austern, welche Noah sich herabließ, mit großem Behagen
zu verschlingen. Eine mehr als gewöhnliche Röte in der Gegend der
Nase des jungen Herrn und ein gewisses Blinzeln seines rechten
Auges verkündigten, daß er ein wenig angetrunken war, und die
besagten Symptome erhielten noch eine Verdeutlichung durch seine
augenscheinliche Begier nach den Austern, die er offenbar hauptsächlich
wegen ihrer kühlenden Eigenschaften bei innerlicher Glut genoß.

«Da ist 'ne prächtige, fette, Noah!» sagte Charlotte. «Die mußt du
probieren.»

«Wie wundervoll doch Austern schmecken!» bemerkte Noah; «und wie schade
ist's, daß man sich immer unbehaglich fühlt, wenn man sie in einiger
Menge genossen hat.»

«'s ist wirklich grausam und unrecht», sagte Charlotte. «Hier ist
wieder 'ne ganz herrliche.»

«Tut mir leid, ich kann nicht mehr. Komm her, Charlotte, daß ich dich
küsse», sagte Noah.

«Wie -- was?» schrie Bumble, hineinstürzend. «Sag das noch einmal,
Bursch!»

Charlotte stieß einen Schrei aus und verbarg ihr Gesicht hinter
der Schürze, während Noah, ohne seine Lage zu verändern, den
Kirchspieldiener mit dem Starrblicke der Trunkenheit angaffte.

«Sag' das noch einmal, du schändlicher, schamloser Schlingel!» fuhr
Bumble fort. «Wie kannst du es wagen, von Küssen zu sprechen? Und Sie,
freches Weibsbild, wie unterstehen Sie sich, ihn dazu aufzumuntern?
Küssen! Pfui!» rief er in starker und gerechter Entrüstung aus.

«Ich wollt' es gar nicht!» sagte Noah bestürzt und flehend. «Sie küßt
mich immer, ich mag es haben wollen oder nicht.»

«O Noah!» rief Charlotte mit einem Blicke des Vorwurfs.

«Ja, es ist wahr,» sprudelte Noah, «du tust's immer. Mr. Bumble, sie
läßt's und läßt's nicht und klopft mich immer unter das Kinn und
flattiert mir auf alle ersinnliche Weise.»

«Schweig!» donnerte Bumble. «Sie packen sich sogleich hinaus, und du,
Musjö Noah, verschließt den Laden und sprichst, bis dein Herr nach
Hause kommt, kein Wort mehr, auf deine eigene Gefahr; und wenn er nach
Hause kommt, so sag ihm, ich ließe ihm sagen, er möchte morgen früh
nach dem Frühstück 'nen Sarg für 'ne alte Frau schicken. Hörst du? --
Küssen! Die Sündhaftigkeit und Gottlosigkeit der geringeren Klasse
in diesem Kirchspielbezirke hat eine schreckliche Höhe erreicht, und
zieht das Parlament ihre Verdorbenheit nicht in Betracht, so ist das
Land zugrunde gerichtet und die Sittlichkeit des Volkes für immer zum
Henker!»

Mit diesen Worten schritt er majestätisch und düster hinaus; und da
wir ihn nun so weit auf seinem Heimwege begleitet und alle nötigen
Anordnungen zum Begräbnisse der alten Frau getroffen haben, wollen wir
uns nach Oliver Twist umsehen und unsere Wißbegier befriedigen, ob er
noch in dem Graben liegt, in welchem Bill Sikes und Toby Crackit ihn
haben liegen lassen.




28. Kapitel.

    Was Oliver nach dem mißlungenen Einbruche begegnete.


«Daß euch die Wölfe zerrissen!» murmelte Sikes zähneknirschend.
«Wollte, daß ich einem von euch nahe genug wäre, er sollte mir erst
Ursache zum Heulen bekommen!»

Indem Sikes mit dem wütendsten Ingrimme, dessen er fähig war, diese
Worte vor sich hin sprach, legte er den verwundeten Knaben über sein
niedergebeugtes Knie und sah sich nach seinen Verfolgern um. Er
vermochte in dem Nebel und der Finsternis nichts zu unterscheiden,
allein desto heller und lauter tönte das Rufen und Schreien der
Nachsetzenden, das Gebell der Hunde rings umher und der Schall der
Lärmglocke durch die Nacht.

«Steh, feiger Schuft!» schrie Sikes Toby Crackit nach, der den
eilfertigsten Gebrauch von seinen langen Beinen zu machen angefangen
hatte, und schon eine Strecke voraus war. «Steh' augenblicklich!»

Toby gehorchte, da er noch nicht vollkommen gewiß war, außer Schußweite
zu sein, und deutlich erkannte, daß Sikes nicht in der Stimmung wäre,
mit sich scherzen zu lassen.

«Hilf mir den Knaben forttragen», tobte der Wütende. «Komm zurück --
hierher!»

Toby kehrte langsam einige Schritte zurück, wagte indes leise und
atemlos einige bescheidene Gegenvorstellungen.

«Geschwinder!» schrie Sikes, legte den Knaben in einen trockenen Graben
und zog eine Pistole hervor. «Hab' mich ja nicht zum Narren!»

Gerade in diesem Augenblick verdoppelte sich der Lärm, und Sikes konnte
erkennen, daß die Verfolger bereits über die Umzäunung des Feldes
kletterten, auf welchem er sich mit Toby und Oliver befand, und daß
ihnen ein paar Hunde mehrere Schritte voraus waren.

«'s ist nichts mehr zu machen, Bill», sagte Toby; «laßt den
Schreiling[AM] und nehmt die Bein' untern Arm!»

  [AM] Knabe.

Er mochte sich lieber der Möglichkeit aussetzen, von dem Freunde
niedergeschossen zu werden, als unfehlbar dem Feinde in die Hände zu
fallen, und rannte daher, so schnell ihn seine Füße tragen wollten,
davon. Sikes biß die Zähne zusammen, warf einen Tuchkragen über den
Knaben, lief an der nächsten Hecke hin, um die Verfolger zu täuschen,
stand vor einer zweiten still, die mit jener in einem rechten Winkel
zusammenstieß, schleuderte seine Pistole hoch in die Luft, wagte einen
verzweifelten Sprung und rannte in einer anderen Richtung als Toby fort.

Seine Eile war unnötig, denn während er über Stock und Block
davoneilte, rief schon einer der drei Nachsetzenden die Hunde zurück,
die gleich ihren Herren kein großes Behagen an der Verfolgung zu finden
schienen und daher augenblicklich gehorchten. Das Kleeblatt war nur
wenige Schritte weit auf das Feld vorgedrungen und stand still, um zu
beraten.

«Mein Rat, oder wenigstens mein Befehl ist der,» sagte der dickste der
drei Männer, «daß wir auf der Stelle umkehren und wieder nach Hause
gehen.»

«Mir ist alles recht, was Mr. Giles recht ist», fiel ein kleinerer Mann
ein, der indes auch keineswegs schmächtig genannt werden konnte und
sehr blaß und sehr höflich war, wie es die Leute häufig sind, wenn die
Furcht sie beherrscht.

«Meine Herren,» nahm der dritte das Wort, der die Hunde zurückgerufen
hatte, «ich möchte nicht gern ungezogen erscheinen. Mr. Giles muß es am
besten wissen.»

«Ja, ja,» fiel der kleinere ein, «was Mr. Giles sagt, dem dürfen wir
nicht widersprechen; nimmermehr, ich kenne meine Stellung Gott sei Dank
zu gut, um mir's herauszunehmen.»

Der kleine Mann schien seine Stellung in der Tat nicht bloß genau
zu kennen, sondern auch sehr unangenehm zu empfinden, denn er stand
zähneklappernd neben den beiden andern.

«Sie fürchten sich, Brittles», sagte Mr. Giles.

«Nicht im mindesten», sagte Brittles.

«Sie fürchten sich allerdings!»

«Sie irren, Mr. Giles.»

«Sie lügen, Brittles.»

Das Zwiegespräch war eine Folge davon, daß Mr. Giles Verdruß empfand,
und sein Verdruß war aus seinem Unwillen darüber entsprungen, daß die
Verantwortlichkeit wegen der Rückkehr nach Hause in der Form eines
Kompliments auf ihn zurückgewälzt worden war. Der dritte Mann beendete
den Streit sehr philosophisch. «Lassen Sie mich Ihnen sagen, wie es
ist», fiel er ein; «wir fürchten uns alle.»

«Sie reden nach Ihrer eigenen Erfahrung», versetzte Mr. Giles, der der
blässeste von den dreien war.

«Allerdings», sagte der Angeredete. «'s ist unter solchen Umständen
ganz natürlich und schicklich, daß man sich fürchtet.»

«Nun, ich fürchte mich auch», sagte Brittles; «aber warum ist es
notwendig, es einem so geradezu in das Gesicht zu sagen?»

Diese offenen Geständnisse besänftigten Mr. Giles, der sofort
einräumte, auch seinerseits einige Furcht zu empfinden, worauf alle
drei in der vollkommensten Einmütigkeit zurückzueilen anfingen. Nicht
lange nachher trug jedoch Mr. Giles, der den kürzesten Atem hatte und
eine große Heugabel trug, auf ein kurzes Verweilen an, um sich wegen
seiner Ausfälle zu entschuldigen.

«Man glaubt es aber gar nicht,» schloß er, «wozu man fähig ist, wenn
einem das Blut warm geworden. Wahrhaftig, ich würde einen Mord begangen
haben -- ich weiß es -- hätten wir einen der Bösewichter gefangen.»

Die anderen beiden hatten ähnliche Überzeugungen und konnten nur nicht
begreifen, wie es zugegangen, daß in ihrer Stimmung eine so plötzliche
Änderung eingetreten war.

«Ich weiß es», sagte Giles; «es kam von der Umzäunung her. Ja, die
Umzäunung des Feldes, auf welchem wir die Halunken fast ertappt hätten,
unterbrach alle Mordgedanken und hemmte die innere Wut. Ich fühlte sie
bei mir im Hinübersteigen vergehen.»

Durch ein merkwürdiges Zusammentreffen hatten die beiden andern
dasselbe Gefühl genau in demselben Augenblick empfunden, so daß Mr.
Giles ganz offenbar recht gehabt hatte, als er sagte: es kam von der
Umzäunung her -- namentlich, da hinsichtlich des Zeitpunktes, in dem
die Veränderung Platz gegriffen hatte, gar kein Zweifel obwalten
konnte, da sich alle drei entsannen, daß sie in dem Augenblicke, da sie
eintrat, die Einbrecher zu Gesicht bekommen hätten.

Dieses Gespräch führten die beiden Männer, welche die Einbrecher
überrascht hatten, und ein wandernder Kesselflicker, der in einem
Nebengebäude geschlafen und sich nebst seinen beiden Hunden hatte
entschließen müssen, an dem gefahrvollen Abenteuer der Diebesverfolgung
teilzunehmen. Mr. Giles diente der alten Dame, welche das Haus
bewohnte, in der doppelten Eigenschaft als Keller- und Haushofmeister,
und Brittles war Bedienter, Gärtner, Ausläufer usw. Die alte Dame
hatte ihn in ihren Dienst genommen, als er noch ein kleiner,
vielversprechender Knabe gewesen war, und er wurde noch immer als ein
solcher behandelt, obgleich er in den Dreißigern stand.

Die drei kühnen Männer setzten unter ermutigenden und die Zeit
kürzenden angenehmen Gesprächen in geschlossener Phalanx ihren Rückzug
fort und bewiesen, obwohl ihnen noch mancher ungewöhnlich starke
Windstoß Schrecken einjagte, die Geistesgegenwart, ihre Laterne
abzuholen, die sie hinter einem Baume hatten stehen lassen, damit sie
den Dieben nicht anzeigte, wohin sie schießen müßten, falls sie etwa
Feuer zu geben geneigt wären.

Sie waren längst zu Hause angelangt, die Luft wurde immer kälter, je
näher der Morgen kam, der Nebel bewegte sich am Boden entlang wie
eine dichte Rauchwolke, das Gras war feucht, die Fußwege und niedrig
gelegenen Stellen waren kotig und schlammig, ein naßkalter Wind
verkündete sein Nahen durch ein hohles Brausen -- und Oliver lag noch
immer bewußtlos in dem Graben, wo ihn Sikes niedergelegt hatte. Im
Osten zeigte sich das erste matte Morgengrauen -- eher dem Tode der
Nacht als der Geburt des Tages zu vergleichen. Die Gegenstände, die in
der Finsternis unheimlich ausgesehen hatten, nahmen immer bestimmtere
Umrisse an und erhielten allmählich ihre gewöhnliche Gestalt. Der Regen
rauschte in Strömen nieder und schlug klatschend auf die entlaubten
Büsche -- aber Oliver empfand kein Ungemach davon; er lag fortwährend
hilflos und ohnmächtig auf seinem harten, feuchten Bette von Erde.

Endlich weckte ihn ein empfindlicher Schmerz; er schrie laut auf und
erwachte. Sein linker, in der Eile mit einem Tuche verbundener Arm
hing schwer und gelähmt an seiner Seite, und das Tuch war mit Blut
getränkt. Er war so schwach, daß er sich kaum in eine sitzende Stellung
emporzurichten vermochte; er blickte matt nach Hilfe umher und ächzte
vor Schmerz. Bebend an allen Gliedern vor Kälte und Erschöpfung, suchte
er sich aufzurichten, fiel aber ohnmächtig der Länge nach wieder nieder.

Eine ihn überfallende Ohnmachtsempfindung, die ihm die Warnung
zuzuraunen schien, er werde unfehlbar sterben müssen, wenn er noch
länger daläge, brachte ihn zum Bewußtsein zurück. Er stand mühsam auf,
ihm schwindelte jedoch, und er wankte gleich einem Betrunkenen von
einer Seite zur anderen. Er hielt sich nichtsdestoweniger aufrecht und
taumelte mit gesenktem Kopfe vorwärts, ohne zu wissen wohin.

In seinem Innern drängten sich ängstigende, verwirrte Gedanken und
Bilder. Es war ihm, als wenn er noch zwischen Sikes und Crackit ginge,
die ergrimmt miteinander zankten; ihre Worte tönten noch in seinen
Ohren, und als ihn ein Fehltritt zum Bewußtsein zurückrief, machte er
die Entdeckung, daß er zu den beiden schrecklichen Männern redete,
als wenn er sich noch in ihrer Gewalt befände. Dann kam es ihm wieder
vor, als wenn er mit Sikes allein wäre, der ihm seine Rolle bei dem
beabsichtigten Einbruche einzuprägen suchte. Unbestimmte, düstere
Gestalten schwebten an ihm hin und wieder, er schreckte zusammen bei
dem vermeintlichen Knalle eines Feuergewehrs, er hörte lautes Rufen
und Schreien, vor seinen Augen flimmerten und verschwanden Lichter,
es summte ihm vor den Ohren, alles vor Verwirrung und Lärmen, er
fühlte sich durch eine unsichere Hand fortgetragen; und während er
so halb wachend träumte, peinigte und ängstigte ihn fortwährend ein
undeutliches Schmerzgefühl, ein Halbbewußtsein seiner unsäglich
jammervollen Lage.

So wankte er weiter und weiter, fast mechanisch durch Gitter oder
Lücken in den Hecken kriechend, bis er einen Weg erreicht hatte; und
jetzt fing es an so stark zu regnen, daß er wirklich erwachte. Er
blickte umher und sah in geringer Entfernung ein Haus, bis zu welchem
er sich fortschleppen zu können meinte. Die Bewohner desselben hatten
vielleicht Mitleid mit ihm, und wo nicht, so war es doch besser, wie er
dachte, in der Nähe menschlicher Wesen zu sterben als allein auf den
öden Feldern. Er sammelte seine letzten Kräfte, und eilte so rasch er
konnte, dem Hause zu.

Als er näher kam, war es ihm, als wenn er es schon gesehen hätte,
wenn er sich auch seiner einzelnen Teile nicht erinnern konnte. Doch
ach, die Gartenmauer! Und dort an jener Stelle hatte er sich in der
vergangenen Nacht auf die Knie niedergeworfen und die beiden Männer
um Erbarmen angefleht. Es war dasselbe Haus, das sie zu berauben
versucht hatten. Auf ein paar Augenblicke überkam ihn ein Gefühl so
entsetzlichen Schreckens, daß er den Schmerz seiner Wunde vergaß und
nur an Flucht dachte. Flucht! Er war kaum imstande, sich auf den Füßen
zu halten, und hätte er die Kräfte dazu gehabt, wohin hätte er fliehen
sollen? Die Gartentür stand offen, er wankte über den Grasplatz, stieg
mit Mühe die Stufen des Portals vor der Haustür hinan und klopfte
leise; die Kräfte schwanden ihm, und er sank ohnmächtig nieder.

Gerade zu derselben Stunde stärkten sich Mr. Giles, Brittles und der
Kesselflicker nach den Strapazen und Schrecken der Nacht in der Küche
durch ein Schälchen Tee, und was es sonst Gutes gab. Nicht, daß es
Mr. Giles' Gewohnheit gewesen wäre, zu große Vertraulichkeit mit der
niederen Dienerschaft zu pflegen, gegen welche er sich vielmehr der
Regel nach nur mit einer leutseligen Herablassung benahm, die stets an
seine höhere Stellung in der Gesellschaft erinnerte. Allein Todesfälle,
Feuersbrünste und Einbrüche machen alle Menschen gleich, und Mr.
Giles saß mit ausgestreckten Füßen am Herde, hatte den linken Arm auf
den Tisch gestützt, illustrierte mit dem rechten einen genauen und
glühenden Bericht über den nächtlichen Raubanfall, und sein Publikum --
zumal die Köchin und das Hausmädchen -- hörte ihm in atemloser Spannung
zu.

«Es mochte halb zwei Uhr sein,» sagte Mr. Giles, «indes kann ich nicht
darauf schwören, ob's nicht dreiviertel war, als ich aufwachte, mich
im Bett herumdrehte, ungefähr so» (er drehte sich bei diesen Worten
auf seinem Stuhle herum und zog den Zipfel des Tischtuches über die
Schultern, um bildlich desto lebhafter die Vorstellung einer Bettdecke
hervorzurufen), «und ein Geräusch zu hören glaubte.»

Hier erblaßte die Köchin und forderte das Hausmädchen auf, die
Küchentür zu verschließen; das Hausmädchen hieß es Brittles, der es den
Kesselflicker hieß, der sich stellte, als ob er nicht hörte.

«Ein Geräusch zu hören glaubte», wiederholte Mr. Giles. «Ich dachte
zuerst bei mir selbst: 's ist eine Täuschung! und legte mich schon
wieder zum Einschlafen zurecht, als ich das Geräusch abermals und
vollkommen deutlich vernahm.»

«Was war es denn für eine Art von Geräusch?» fragte die Köchin.

«Ein krachendes», erwiderte Mr. Giles.

«Ich denke, es war mehr, wie wenn eine eiserne Stange auf einem
Reibeisen gerieben wird», fiel Brittles ein.

«So war es, als *Sie* es hörten», sagte Giles; «zu der Zeit aber, als
ich es vernahm, hatte es einen krachenden Ton. Ich warf die Bettdecke
zurück» (er wiederholte die Bewegung mit dem Tischtuche), «richtete
mich zum Sitzen empor und horchte.»

Die Köchin und das Hausmädchen riefen zugleich aus: «Daß sich Gott
erbarm'!» und rückten zusammen.

«Ich hörte es so deutlich, als wenn es dicht vor meinem Bette wäre»,
fuhr Giles fort, «und dachte: Da wird eine Tür oder ein Fenster
aufgebrochen. Was ist zu tun? Ich will den guten Jungen Brittles
wecken und ihn retten, damit er nicht in seinem Bette ermordet wird;
tu' ich's nicht, so kann ihm die Kehle vom rechten bis zum linken Ohre
abgeschnitten werden, ohne daß er es merkt.»

Hier richteten sich aller Blicke auf Brittles, der die seinigen auf den
Redner heftete und ihn mit weit geöffnetem Munde anstarrte, während
seine Mienen den grenzenlosesten Schrecken ausdrückten.

«Ich stieß die Bettdecke von mir,» sprach Giles, das Tischtuch von sich
schleudernd und die Köchin und das Hausmädchen scharf fixierend, «stieg
leise aus dem Bette, zog --»

«Es sind Damen anwesend, Mr. Giles!» murmelte der Kesselflicker.

«-- Meine Pantoffel an, Sir,» fuhr Giles, sich zu ihm wendend und
den stärksten Nachdruck auf seine Pantoffel legend, fort, «nahm
die geladene Pistole zur Hand, die immer mit dem Silberzeugkorbe
hinaufgebracht wird, ging auf den Zehen in Brittles Kammer und sagte,
sobald ich ihn aus dem Schlafe gerüttelt hatte: >Brittles, erschrecken
Sie nicht!<»

«Ja, das sagten Sie, Mr. Giles», fiel Brittles mit bebender Stimme ein.

«>Brittles, ich glaube, wir sind verloren<, sagt' ich,» fuhr Giles
fort, «>aber seien Sie nur ohne Furcht.<»

«Zeigte er denn auch keine Furcht?» fragte die Köchin.

«Nein», antwortete Giles; «er war so unverzagt -- fast so unverzagt wie
ich selber.»

«Ich wäre auf der Stelle gestorben, wenn ich's gewesen wäre», bemerkte
das Hausmädchen.

«Sie sind ein junges Mädchen», fiel Brittles ein, der ziemlich herzhaft
zu werden anfing.

«Brittles hat recht», sagte Giles mit beiläufigem Kopfnicken; «von
einem Mädchen war nichts anderes zu erwarten. Wir aber, als Männer,
nahmen eine Blendlaterne aus Brittles' Kammer und fühlten uns in der
Pechrabenfinsternis hinunter.» (Er war aufgestanden, hatte die Augen
geschlossen, tappte ein paar Schritte vorwärts und durchsägte, um seine
Schilderung mit angemessener Aktion zu begleiten, mit den Armen die
Luft, bis er mit der Köchin in eine unangenehme Berührung kam und die
Köchin und das Hausmädchen zu schreien anfingen, worauf er nach seinem
Stuhle zurückeilte.) «Was hat das zu bedeuten?» unterbrach er sich
plötzlich; «es wird geklopft -- öffne jemand die Haustür!»

Niemand regte sich.

«Das ist doch seltsam, daß zu einer so frühen Morgenstunde geklopft
wird», fuhr Giles, umherschauend und bleichen Antlitzes nur bleiche
Gesichter gewahrend, fort; «allein die Tür muß geöffnet werden. He!
Holla! hört denn niemand?»

Mr. Giles richtete bei diesen Worten die Blicke auf Brittles; allein
der von Natur blöde, bescheidene Jüngling hielt sich mutmaßlich
in der Tat für niemand und meinte sicher, daß die Frage unmöglich
an ihn gerichtet sein könne. Jedenfalls gab er keine Antwort. Mr.
Giles sendete dem Kesselflicker auffordernde Blicke zu, allein der
Kesselflicker war urplötzlich eingeschlafen. Die Frauenzimmer kamen
nicht in Betracht.

«Wenn Brittles die Tür etwa lieber in Gegenwart von Zeugen öffnen
will,» sagte Mr. Giles nach einem kurzen Stillschweigen, «so bin ich
bereit, einen solchen abzugeben.»

«Ich auch», sagte der Kesselflicker, ebenso plötzlich wieder
aufwachend, wie er eingeschlafen war.

Brittles kapitulierte auf diese Bedingungen, und als man beim Öffnen
der Fensterläden die Entdeckung machte, daß es heller Tag war, und
dadurch gar sehr an Mut gewann, so zog die kleine, tapfere Schar aus,
die Hunde voran und die Frauenzimmer in der Nachhut. Gemäß dem Rate Mr.
Giles' sprachen alle sehr laut, um dem Feinde sogleich kundzutun, wie
zahlreich sie waren, und gemäß einer unübertrefflichen, von demselben
Gentleman ausgesonnenen Kriegslist wurden auf dem Hausflur die Hunde in
die Schwänze gekniffen, damit sie ein wütendes Bellen erheben möchten,
was sie auch taten.

Nachdem diese Vorsichtsmaßregeln getroffen waren, faßte Mr. Giles den
Kesselflicker fest am Arme (damit er nicht fortliefe, wie Mr. Giles
scherzend sagte) und gab den Befehl, die Tür zu öffnen. Brittles
gehorchte. Einer blickte dem anderen bebend über die Schultern, und
die Schar gewahrte nichts Fürchterlicheres als den armen, kleinen
Oliver Twist, der bleich und erschöpft die Augen aufschlug und stumm um
Mitleid flehte.

«Ein Knabe!» rief Mr. Giles, den Kesselflicker mutig zurück- und sich
selber vordrängend, aus. «Was ist das -- wie -- Brittles -- erkennen
Sie ihn?»

Brittles, der beim Öffnen der Tür hinter dieselbe getreten war, stieß
einen Schrei des Wiedererkennens aus, sobald er Oliver erblickte. Giles
faßte den Knaben bei einem Beine und einem Arme -- zum Glücke nicht
bei dem verwundeten -- zog ihn herein und legte ihn der Länge nach auf
die Steinplatten nieder. «Hier, hier!» schrie Mr. Giles in größter
Erregtheit die Treppe hinauf, «hier ist einer von den Dieben, Ma'am!
Wir haben einen Dieb, Miß -- einen Verwundeten, Miß! Ich traf ihn, Miß,
und Brittles hielt das Licht!»

«In einer Laterne, Miß!» schrie Brittles, eine Hand an den Mund
haltend, damit sein Ruf desto sicherer hinaufdränge.

Die Köchin und das Hausmädchen liefen hinauf, um der Herrschaft
zu verkündigen, daß Mr. Giles einen Räuber gefangen habe, und der
Kesselflicker bemühte sich, Oliver zum Bewußtsein zurückzubringen,
damit er nicht stürbe, bevor er gehängt würde. Nach einiger Zeit
ertönte von oben durch den Lärm eine sanfte und wohlklingende, einer
jungen Dame angehörende Stimme: «Giles, Giles!»

«Hier, Miß, hier bin ich! Erschrecken Sie nicht, Miß; ich habe keinen
bedeutenden Schaden genommen! Er leistete keinen sehr verzweifelten
Widerstand, Miß; ich überwältigte ihn sehr bald.»

«Still doch; Sie erschrecken ja meine Tante fast ebensosehr, wie es die
Diebe selbst getan haben. Ist der arme Mensch stark beschädigt?»

«Er hat eine furchtbare Wunde, Miß», rief Giles mit unbeschreiblichem
Wohlbehagen hinauf.

«Er sieht aus, als wenn er den Geist aufgeben will, Miß», schrie
Brittles, wie zuvor eine Hand an den Mund haltend. «Wollen Sie nicht
herunterkommen, Miß, und ihn sehen, falls er --»

«So schreien Sie doch nicht so entsetzlich. Seien Sie einen Augenblick
still; ich will mit meiner Tante sprechen.»

Die Sprecherin eilte mit leisen Fußtritten fort, kehrte bald wieder
zurück und erteilte den Befehl, den Verwundeten vorsichtig hinauf in
Mr. Giles' Zimmer zu tragen; Brittles sollte sogleich den Pony satteln,
nach Chertsey reiten und eiligst einen Konstabler und den Doktor holen.

«Wollen Sie ihn aber nicht erst einmal sehen, Miß?» rief Giles mit so
viel Stolz, wie wenn Oliver ein seltener und prachtvoller Vogel wäre,
den er heruntergeschossen hätte.

«Nicht um die Welt!» erwiderte die junge Dame. «Der arme Mensch! Giles,
behandeln Sie ihn ja recht gut, und wenn es auch nur um meinetwillen
wäre!»

Der alte Diener des Hauses blickte, als sie sich entfernte, zu ihr
hinauf, so stolz und wohlgefällig, als ob sie sein eigenes Kind
gewesen wäre, und half sodann Oliver mit der liebevollen Sorgfalt und
Aufmerksamkeit einer Frau hinauftragen.




29. Kapitel.

    Von den Bewohnern des Hauses, in welchem Oliver sich befand.


In einem artigen Zimmer -- dessen Mobilien freilich mehr nach
altmodischer Bequemlichkeit als nach moderner Eleganz aussahen --
saßen zwei Damen an einem wohlbesetzten Frühstückstische. Mr. Giles
wartete im vollständigen schwarzen Anzuge auf. Er stand kerzengerade
in der Mitte zwischen dem Schenk- und Frühstückstische mit
zurückgeworfenem und fast unmerklich zur Seite geneigtem Kopfe, den
linken Fuß vorangestellt und mit der rechten Hand im Busen, während
die herunterhängende Linke einen Präsentierteller hielt, und sah aus,
als wenn er sich des angenehmen Bewußtseins seiner Verdienste und
Wichtigkeit freute.

Die eine der beiden Damen war betagt, allein die hohe Lehne ihres
Stuhles war nicht gerader als ihre Haltung. Ihr Anzug war ein Muster
von Sauberkeit und Genauigkeit, altmodisch, doch nicht ohne Spuren der
Einwirkung des Tagesgeschmacks. So saß sie stattlich da, die gefalteten
Hände auf dem Tische vor ihr, die Augen -- denen die Jahre nur wenig
von ihrem Glanze genommen -- aufmerksam auf die jüngere Dame geheftet,
die in der ersten zarten Blüte der Weiblichkeit stand, eine der
jungfräulichen Gestalten, von welchen wir ohne Sünde annehmen mögen,
daß Engel sie bewohnen, wenn der Allmächtige zur Ausführung seiner
Absichten jemals zuläßt, daß sich die Himmelsbewohner in Gestalten der
Sterblichen verkörpern dürfen.

Sie befand sich noch im siebzehnten Jahre, und ihre Figur war so leicht
und ätherisch, so zart und edel, so lieblich und schön, als wäre die
Erde ihre Wohnstätte nicht, als könnten die gröberen Wesen dieser Welt
keine zu ihr passende Mitgeschöpfe sein. Der Geist, der aus ihren
dunkelblauen Augen leuchtete und aus ihren edlen Zügen sprach, schien
ihrem Alter zuvorgeeilt und kaum von dieser Welt zu sein; und doch
verkündete der lebensvolle, freundlich-holde Ausdruck ihrer Mienen,
die tausend Lichter, die auf ihrem rosigen Antlitz spielten und keinen
Schatten auf ihm lagern ließen, ihr Lächeln -- ihr frohes, seliges
Lächeln -- die höchste Gesinnungsschöne, den reinsten Herzensadel, die
wärmste Liebe und Zärtlichkeit, die besten Gefühle und Eigenheiten der
menschlichen Natur. Ihr Lächeln, ihr heiteres, glückstrahlendes Lächeln
war für häuslichen Frieden, häusliches Glück geschaffen.

Sie war eifrig mit den kleinen Anordnungen zum Frühstück beschäftigt,
und als sie zufällig die Augen aufschlug, während die ältere Dame
sie anblickte, strich sie freundlich ihr einfach auf der Stirn
gescheiteltes Haar zurück, und aus ihren Blicken leuchtete eine solche
tief-innige Zärtlichkeit und natürlich-ungefälschte Liebenswürdigkeit
hervor, daß selige Geister gelächelt haben möchten, sie so zu schauen.

Die ältere Dame lächelte, doch ihr Herz war schwer, und sie trocknete
eine Zähre in dem freundlichen Auge ab.

«Brittles ist also schon seit einer Stunde fort?» fragte sie nach einem
kurzen Stillschweigen.

«Eine Stunde und zwölf Minuten, Ma'am», antwortete Giles, auf seine
silberne Uhr blickend, die er an einem schwarzen Bande herauszog.

«Er ist immer langsam», bemerkte die alte Dame.

«Er war von jeher ein langsamer Bursche, Ma'am», sagte Giles, und in
Anbetracht dessen, daß Brittles, beiläufig gesagt, einige dreißig Jahre
ein langsamer Bursche gewesen war, war es nicht eben wahrscheinlich,
daß er jemals ein hurtiger werden würde.

«Ich glaube, er wird eher schlimmer als besser», fuhr die alte Dame
fort.

«Es würde gar nicht zu entschuldigen sein, wenn er sich aufhielte, um
etwa mit anderen Knaben zu spielen», fiel die junge Dame lächelnd ein.

Mr. Giles überlegte offenbar, ob er sich mit Schicklichkeit auch ein
ehrerbietiges Lächeln erlauben dürfe, als ein Gig vorfuhr, ein dicker
Herr heraussprang, in das Haus hereinstürmte und so eilig in das Zimmer
hereinpolterte, daß er fast Mr. Giles und den Teetisch umgeworfen hätte.

«So etwas ist mir ja in meinem ganzen Leben nicht vorgekommen!» rief
er aus. «Meine beste Mrs. Maylie -- daß sich der Himmel erbarme --
und obendrein in der Stille der Nacht -- es ist ganz unerhört, ganz
unerhört!» Er schüttelte bei diesen Beileidsbezeigungen beiden Damen
die Hände, nahm Platz und erkundigte sich nach ihrem Befinden. -- «'s
ist ein Wunder, daß der Schreck Sie nicht getötet -- auf der Stelle
getötet hat!» fuhr er fort. «In aller Welt, warum schickten Sie nicht
zu mir? Wahrhaftig, mein Bedienter hätte in einer Minute hier sein
sollen oder ich selbst und mein Assistent -- jedermann würde mit
Freuden herbeigeeilt sein. Es versteht sich ja ganz von selbst -- unter
solchen Umständen -- Himmel! -- und so unerwartet -- und in der Stille
der Nacht!»

Der Doktor schien besonders durch den Umstand ganz außer sich geraten
zu sein, daß der Einbruch unerwartet und zu nächtlicher Zeit versucht
worden war, als wenn es die feststehende Gewohnheit der im Fache des
Einbrechens arbeitenden Gentlemen wäre, ihre Geschäfte um Mittag
abzumachen und ihr Erscheinen ein paar Tage vorher durch die Briefpost
anzukündigen.

«Und Sie, Miß Rose», sagte der Doktor zu der jungen Dame; «ich --»

«Ich befinde mich vortrefflich», unterbrach sie ihn; «aber oben liegt
ein Verwundeter, und die Tante wünscht, daß Sie ihn besuchen.»

«Ah, ich entsinne mich», versetzte der Doktor. «Wie ich höre, haben Sie
ihm die Wunde beigebracht, Giles.»

Mr. Giles, der in einem Fieber von Aufregung die Tassen geordnet hatte,
errötete sehr stark und erwiderte, daß er die Ehre habe.

«Die Ehre?» sagte der Doktor. «Doch mag sein, daß es ebenso ehrenvoll
ist, einen Dieb in einem Waschhause, wie einen Gegner auf zwölf
Schritte weit zu treffen. Bilden Sie sich ein, er hätte in die Luft
geschossen und Sie haben ein Duell gehabt, Giles.»

Mr. Giles, der in dieser scherzhaften Behandlung der Sache einen
ungerechten Versuch erblickte, seinen Ruhm zu verkleinern, erwiderte
ehrerbietig, daß es seinesgleichen nicht zukäme, ein Urteil darüber
auszusprechen, allein er lebe doch des Glaubens, daß die Sache für den
Getroffenen kein Spaß gewesen sei.

«Beim Himmel, das ist wahr!» sagte der Doktor. «Wo ist er? Führen Sie
mich zu ihm. Ich werde bald wieder bei Ihnen sein, Mrs. Maylie. Das
ist das kleine Fenster, durch das er eingestiegen ist? Es ist kaum zu
glauben.»

Er folgte, fortwährend sprechend, Mr. Giles die Treppe hinauf, und
während er hinaufgeht, sei dem Leser gesagt, daß Mr. Losberne, der
auf zehn Meilen im Umkreise unter dem Namen des «Doktors» bekannte
Wundarzt, mehr infolge eines heiteren Temperaments als guten Lebens
beleibt geworden und ein so gutherziger und biederer, nebenher auch
wunderlicher alter Junggeselle war, daß man in einem fünfmal größeren
Umkreise kaum seinesgleichen finden dürfte.

Der Doktor blieb weit länger fort, als es die Damen vermutet hatten. Es
wurde ein langer, flacher Kasten aus dem Gig geholt, häufig geklingelt,
die Dienerschaft lief treppauf, treppab; mit einem Worte, es mußte
wohl etwas Wichtiges vorgehen. Endlich trat er mit einer äußerst
geheimnisvollen Miene wieder herein, verschloß die Tür sorgfältig und
sagte, während er mit dem Rücken an sie gelehnt stehenblieb, als wenn
er verhindern wollte, daß jemand hereinkäme: «Mrs. Maylie, dies ist ein
ganz wunderbarer Fall.»

«Ich will doch hoffen, daß der Patient sich nicht in Gefahr befindet?»
fragte die alte Dame.

«Es würde den Umständen nach nicht zu verwundern sein,» erwiderte
Losberne, «obwohl ich es nicht glaube. Haben Sie den Dieb gesehen?»

«Nein.»

«Auch sich ihn nicht beschreiben lassen?»

«Nein.»

«Bitt' um Vergebung, Ma'am», fiel Giles ein; «ich wollte Ihnen eben
eine Beschreibung von ihm geben, als Doktor Losberne erschien.»

Die Sache verhielt sich indes so, daß sich Mr. Giles nicht hatte
überwinden können, das Geständnis zu machen, daß er nur einen Knaben
getroffen habe. Er hatte wegen seines mutvollen Benehmens so große
Lobsprüche erhalten, daß er nicht umhin gekonnt, die Aufhellung der
Sache noch ein paar entzückende Minuten aufzuschieben, um noch ein
Weilchen in dem süßen Bewußtsein des Ruhmes einer unerschütterlichen
Herzhaftigkeit zu schwelgen.

«Rose wünschte den Mann zu sehen,» sagte Mrs. Maylie, «allein ich
wollte nichts davon hören.»

«Hm!» sagte der Doktor. «Er sieht aber nicht eben sehr fürchterlich
aus. Möchten Sie ihn auch nicht in meiner Gegenwart sehen?»

«Warum nicht, wenn Sie es für notwendig halten?» erwiderte die alte
Dame.

«Ich muß es für notwendig erklären oder bin doch jedenfalls überzeugt,
daß Sie es gar sehr bedauern würden, es nicht getan zu haben, wenn Sie
ihn später zu sehen bekämen. Er ist vollkommen ruhig, und wir haben
auch in allen Beziehungen für ihn gesorgt. Erlauben Sie mir Ihren Arm,
Miß Rose. Auf meine Ehre, Sie brauchen nicht im mindesten Furcht zu
hegen.»




30. Kapitel.

    Was die beiden Damen Maylie und Doktor Losberne von Oliver denken.


Der Doktor legte unter noch viel anderen redseligen Versicherungen, daß
die Damen durch den Anblick des Verbrechers angenehm überrascht werden
würden, den Arm der jüngeren in den seinigen, bot Mrs. Maylie seine
andere freie Hand und führte sie mit der förmlichsten Galanterie die
Treppe hinauf.

«Lassen Sie mich nun hören, was Sie von ihm denken», sagte er, als sie
vor der Tür des Patienten standen. «Er hat sich seit vielen Tagen den
Bart nicht abnehmen lassen, sieht aber trotzdem keineswegs wie ein
Gurgelabschneider aus.»

Er führte die Damen hinein und an das Bett, schob die Vorhänge zurück,
und sie erblickten statt eines grimmig aussehenden Banditen, den
sie zu sehen erwartet hatten -- einen vor Schmerz und Erschöpfung
eingeschlafenen Knaben. Olivers verbundener Arm lag auf seiner Brust,
und sein Kopf ruhte auf dem andern, der durch sein langes, wallendes
Haar fast versteckt war. Rose setzte sich, während Losberne im
Anschauen des Knaben verloren dastand, oben an das Bett des letzteren,
beugte sich über ihn und strich ihm leise das Haar von der Stirn, auf
welche ein paar Tränen aus ihrem Auge herabfielen.

Der Knabe bewegte sich und lächelte im Schlafe, als wenn ihn diese
Zeichen des Mitgefühls und zarten Erbarmens in einen süßen Traum
von nie gekannter Liebe und Zärtlichkeit versenkt hätten, so wie
entfernte Töne einer lieblichen Melodie oder das Rauschen des Wassers
an einem heimlichen Plätzchen oder der Duft einer Blume oder selbst
das Aussprechen eines teuren Namens bisweilen plötzlich unbestimmte
Bilder in diesem Dasein nie erlebter Szenen, die gleich einem Hauche
wieder verschwinden, vor die Seele zaubert, Szenen, die aus der dunklen
Erinnerung eines längst vergangenen glücklichen Daseins emporzutauchen
scheinen, denn keine Kraft der menschlichen Seele vermag es, sie wieder
zurückzurufen.

«Ich bin fast außer mir vor Verwunderung», flüsterte die alte Dame.
«Dieses arme Kind kann nun und nimmermehr ein Diebes- und Räuberzögling
sein.»

«Das Laster schlägt seinen Wohnsitz in gar vielerlei Tempeln auf»,
versetzte Losberne seufzend, indem er den Vorhang wieder fallen ließ,
«und erscheint oft genug in lieblicher Gestalt.»

«Aber doch nicht bei solcher Jugend», fiel Rose ein.

«Meine teure Miß,» entgegnete der Wundarzt mit traurigem Kopfschütteln,
«das Verbrechen beschränkt sich gleich dem Tode nicht auf die Bejahrten
und Abgelebten allein. Die Jugendlichsten und Schönsten sind nur zu oft
seine auserwählten Opfer.»

«O Sir, können Sie wirklich glauben, daß dieser zarte Knabe sich
freiwillig den schlimmsten Bösewichtern zugesellt hat?» wandte Rose
lebhaft ein.

Losberne schüttelte den Kopf mit einer Miene, als ob er es für sehr
möglich hielte, und führte die Damen in das anstoßende Zimmer, damit
der kleine Patient, wie er sagte, nicht gestört würde.

«Aber selbst, wenn er ruchlos gewesen wäre,» fuhr Rose fort, «so
bedenken Sie, wie jung er ist; daß er vielleicht nie eine liebevolle
Mutter, vielleicht nicht einmal ein elterliches Haus gekannt hat, und
wie wahrscheinlich es ist, daß ihn schlechte Behandlung, Schläge oder
Hunger genötigt haben, sich an Menschen anzuschließen, die ihn zum
Verbrechen zwangen. Tante, beste Tante, bedenken Sie das doch ja, ehe
Sie zugeben, daß der kranke Kleine in ein Gefängnis geschleppt wird,
das auf alle Fälle das Grab jeder Hoffnung der Besserung bei ihm sein
würde. Oh, so gewiß Sie mich liebhaben und wissen, daß ich bei Ihrer
Güte und Zärtlichkeit meine Elternlosigkeit nie empfunden, daß ich sie
aber schmerzlich hätte fühlen können und gleich hilf- und schutzlos wie
dies arme Kind sein könnte, haben Sie Mitleid mit ihm, ehe es zu spät
ist.»

«Mein liebes Kind,» sagte die ältere Dame, das weinende Mädchen an
die Brust drückend, «glaubst du, ich würde auch nur ein Haar seines
Hauptes krümmen lassen wollen?»

«O nein, nein, beste Tante, Sie wollen und könnten es nicht!» rief Rose
mit Lebhaftigkeit aus.

«Nein, sicherlich nicht,» fuhr Mrs. Maylie mit bebender Lippe fort,
«meine Tage neigen sich ihrem Ende zu, und möge ich Barmherzigkeit
erfahren, wie ich sie anderen erweise. Was kann ich zur Rettung des
Knaben tun, Sir?»

«Lassen Sie mich nachdenken, Ma'am,» erwiderte der Doktor, «lassen Sie
mich nachdenken.»

Mr. Losberne steckte seine Hände in die Taschen und ging einigemal im
Zimmer auf und nieder, stand dann wieder still, wiegte sich auf seinen
Fußspitzen, rieb heftig die Stirn und sagte endlich: «Ich hab's, Ma'am.
-- Ja -- ich sollte meinen, daß ich es schon einrichten könnte, wenn
Sie mir unbeschränkte Vollmacht geben wollen, Giles und Brittles, den
großen Jungen, in das Bockshorn zu jagen. Giles ist ein alter Diener
Ihres Hauses und ein treuer Mensch, das weiß ich; und Sie können es
bei ihm auf tausenderleiweise wieder gutmachen und ihn obendrein dafür
belohnen, daß er ein so guter Schütze ist. Sie haben doch nichts
dawider?»

«Wenn es kein anderes Mittel gibt, das Kind zu retten, nein»,
antwortete Mrs. Maylie.

«Auf mein Wort, es gibt kein anderes Mittel», versicherte Losberne.

«Dann bekleidet Tante Sie mit Vollmacht», sagte Rose, durch Tränen
lächelnd; «aber bitte, setzen Sie den beiden guten Leuten nicht härter
zu, als es unumgänglich notwendig ist.»

«Sie scheinen zu glauben,» entgegnete der Doktor, «daß alle Welt heute
zu Hartherzigkeit geneigt ist, Sie selbst allein ausgenommen, Miß
Rose. Ich will nur um des aufwachsenden männlichen Geschlechts willen
insgeheim hoffen, daß der erste Ihrer würdige junge Mann, der Ihr
Mitleid in Anspruch nimmt, seine Werbung bei Ihnen anbringt, wenn Sie
sich in einer ebenso verwundbaren und weichherzigen Stimmung befinden,
und wünschte nichts mehr, als daß ich selbst ein junges Herrlein sein
möchte, um sogleich einen so günstigen Augenblick wie den gegenwärtigen
benutzen zu können.»

«Sie sind ein ebenso großer Knabe wie unser guter Brittles», sagte Rose
errötend.

«Dazu gehört eben nicht viel», versetzte der Doktor herzlich lachend.
«Doch um auf den kleinen Knaben zurückzukommen: wir haben die
Hauptsache bei unserem Vertrage noch nicht erwähnt. Er wird ohne
Zweifel in ungefähr einer Stunde aufwachen, und obgleich ich dem
breitmäuligen Konstabler unten gesagt habe, daß bei Gefahr seines
Lebens nicht mit ihm gesprochen werden dürfe, so denke ich doch, daß
wir es ganz dreist tun können. Ich mache nun die Bedingung -- daß ich
ihn in Ihrer Gegenwart examiniere, und daß er, wenn wir seinen Aussagen
nach urteilen müssen, und wenn ich Ihnen zur Befriedigung Ihres kalten
Verstandes dartun kann, daß er (was mehr als möglich) durch und durch
verderbt ist, seinem Schicksale ohne weitere Einmischung -- zum
wenigsten von meiner Seite -- überlassen wird.»

«Nein, Tante, nein!» flehte Rose.

«Ja, Tante, ja!» sagte der Doktor. «Sind wir einig?»

«Er kann nicht im Laster verhärtet sein», sagte Rose; «es ist
unmöglich.»

«Desto besser», entgegnete Losberne; «dann ist um so mehr Grund
vorhanden, meinen Vorschlag dreist anzunehmen.»

Der Vertrag wurde endlich geschlossen, und man setzte sich, um in
großer Spannung Olivers Erwachen abzuwarten.

Die Geduld der beiden Damen sollte indes auf eine längere Probe
gestellt werden, als sie nach des Doktors Äußerungen gefürchtet hatten,
denn eine Stunde verging nach der andern, und Oliver lag fortwährend im
festesten Schlummer. Es war Abend geworden, als ihnen der gutherzige
Losberne die Nachricht brachte, daß der Patient endlich hinreichend
wach geworden sei, um Rede und Antwort stehen zu können. Er sei sehr
krank, wie Losberne sagte, und sehr schwach infolge des Blutverlustes,
allein sein Gemüt, durch den Wunsch, etwas zu enthüllen, so beunruhigt,
daß es unbedingt besser sei, ihn reden zu lassen, als -- was sonst
geschehen sein würde -- darauf zu bestehen, daß er sich bis zum
folgenden Morgen ruhig verhalten solle.

Die Unterredung dauerte lange, denn Oliver erzählte ihnen seine ganze
Lebensgeschichte, und oft nötigten ihn Schmerz oder Erschöpfung,
innezuhalten. Die schwache Stimme des kranken Knaben, sein rührend
schauerlicher Bericht über eine lange Reihe trostloser Leiden und
Mißgeschicke, von verhärteten Menschen über ihn verhängt, hörte sich in
dem verdunkelten Zimmer gar feierlich an. Oh, wieviel weniger Unrecht
und Ungerechtigkeit, Leid und Grämen, Grausamkeit und Elend, wie es
jeder Tag mit sich bringt, würde es auf dieser Welt geben, wenn wir --
während wir unsere Mitmenschen unterdrücken und quälen -- nur mit einem
einzigen Gedanken an die finster drohenden Anklagen gegen uns dächten,
die gleich dichten, schweren Wolken freilich langsam, aber desto
gewisser zum Himmel emporsteigen, um dereinst ihre Racheblitze auf
unsere Häupter herabzusenden -- wenn wir im Geist nur einen Augenblick
hören wollten auf das schauerliche Zeugnis der Stimmen der Toten und
zu ihrem und unserem Schöpfer und Richter Hinübergegangenen, die keine
menschliche Macht oder Gewalt unterdrücken, kein Stolz verstummen
machen kann!

Olivers Kissen war in dieser Nacht durch Frauenhände geglättet, und
Liebenswürdigkeit und Tugend bewachten seinen Schlummer. Er empfand
eine selige Ruhe und hätte sterben mögen ohne Murren.

Sobald die Unterredung mit ihm beendet und er, was fast augenblicklich
geschah, wieder eingeschlummert war, trocknete der Doktor seine Augen,
verwünschte sie wie gewöhnlich wegen ihrer Schwäche und begab sich
darauf in die Küche hinunter, um seinen Feldzug gegen Mr. Giles und
Konsorten zu beginnen. Er fand die ganze Dienerschaft nebst dem
Konstabler und dem Kesselflicker versammelt, der in Anbetracht seiner
geleisteten Dienste eine besondere Einladung erhalten hatte, den
ganzen Tag zu bleiben und sich wieder zu stärken und zu erquicken. Der
Konstabler war ein Gentleman mit einem großen Stabe, großem Kopfe,
großem Munde und großen Halbstiefeln und sah aus, als wenn er sehr
reichlich im gespendeten Ale gezecht hätte, was auch in der Tat der
Fall war. Als der Doktor eintrat, wurden noch immer die Abenteuer der
vergangenen Nacht besprochen, Mr. Giles verbreitete sich über seine
Geistesgegenwart, und Brittles bekräftigte, mit einem Alekrug in der
Hand, alles, was Mr. Giles erst noch sagen wollte.

«Bleibt sitzen», sagte der Doktor mit einer Handbewegung.

«Schönen Dank, Sir», sagte Mr. Giles. «Misses befahlen mir, ein wenig
Ale auszuteilen, und da es mir in meinem eignen kleinen Zimmer zu eng
war, und da mich nach Gesellschaft verlangte, so trinke ich meinen
Anteil hier.»

Brittles und die übrigen drückten durch ein leises Gemurmel ihr
Vergnügen über Mr. Giles' Herablassung aus, und Mr. Giles blickte mit
einer Gönnermiene umher, welche deutlich sagte, daß er, solange sie
ein schickliches Benehmen beobachteten, ihre Gesellschaft sicher nicht
verlassen würde.

«Wie befindet sich der Patient heute abend, Sir?» fragte Giles.

«Nicht eben gar zu gut», erwiderte der Doktor. «Ich fürchte, Giles, daß
Sie sich selbst in eine arge Klemme gebracht haben.»

«Ich will doch hoffen, Sir, Sie wollen nicht sagen, daß er sterben
werde», sagte Giles zitternd. «Ich könnte nie wieder ruhig werden, wenn
es geschähe. Sir, ich möchte um alles Silberzeug im Lande keinem Knaben
das Leben nehmen, nicht einmal Brittles.»

«Das ist nicht der Kernpunkt der Sache», fuhr der Doktor geheimnisvoll
fort. «Fürchten Sie Gott, und haben Sie ein Gewissen, Giles?»

«Ja, Sir, ich sollte meinen», stotterte der sehr blaß gewordene
Haushofmeister.

«Und wie steht es mit Ihnen, junger Mensch -- haben Sie auch ein
Gewissen, Brittles?»

«Barmherziger Himmel, Sir -- wenn Mr. Giles ein Gewissen hat, hab' ich
auch eins.»

«Dann sagt mir beide -- alle beide: wollt ihr es auf euer Gewissen
nehmen, zu beschwören, daß der verwundete, oben liegende Knabe derselbe
ist, der gestern nacht durch das kleine Fenster gesteckt wurde? Heraus
mit der Sprache! Sagt an, sagt an!»

Der Doktor, der aller Welt als der sanftmütigste Mann von der Welt
bekannt war, sprach diese Worte in einem so schauerlich-zornigen Tone,
daß Giles und Brittles, die durch Ale und Aufregung ziemlich außer
Fassung waren, einander vollkommen betäubt anstarrten. -- «Achten Sie
auf die Antwort, welche erfolgen wird, Konstabler», sprach der Doktor
weiter und hob mit großer Feierlichkeit den Zeigefinger empor; «es kann
früher oder später viel darauf ankommen.»

Der Konstabler nahm eine so weise Miene an, wie er nur konnte, und
griff zu seinem Stabe.

«Sie werden bemerken, es handelt sich einfach um die Identität der
Person», fuhr der Doktor fort.

«Sie haben vollkommen recht, Sir», sagte der Konstabler unter heftigem
Husten, denn er hatte rasch seinen Krug geleert, wovon ihm etwas in die
unrechte Kehle gekommen war.

«Es wird in das Haus eingebrochen,» sagte der Doktor, «und zwei Leute
sehen einen Knaben auf einen einzigen flüchtigen Augenblick, mitten
im Pulverdampfe, in der Verwirrung des nächtlichen Schreckens und
Aufruhrs. Am folgenden Morgen kommt ein Knabe in dieses Haus, und weil
er zufällig den Arm verbunden hat, legen die Leute gewaltsam Hand an
ihn, bringen sein Leben dadurch in die augenscheinlichste Gefahr und
schwören, daß er an dem Einbruch teilgenommen habe. Die Frage ist nun
die, ob das Verhalten besagter Leute durch die Umstände gerechtfertigt
erscheint, und wo nicht, in was für eine Lage sie sich selber
versetzen? Und nun noch einmal,» donnerte der Doktor, während der
Konstabler Giles und Brittles mit bedenklich-mitleidiger Miene ansah,
«seid ihr gewillt und imstande, vor Gott und auf das heilige Evangelium
die Identität des Knaben zu beschwören?»

Brittles blickte Giles und Giles Brittles zweifelhaft und fragend an;
der Konstabler hielt die Hand hinter das Ohr, damit ihm ja nichts
von der Antwort entgehen möchte; die Köchin, das Hausmädchen und der
Kesselflicker beugten sich vor, um zu lauschen, und der Doktor schaute
mit scharfen Blicken umher, als das Heranrollen eines Wagens und gleich
darauf das Läuten der Gartentorglocke vernommen wurde.

«Es sind die Polizeimänner aus London», rief Brittles, sich sehr
erleichtert fühlend, aus.

«In aller Welt, wie kommen denn die hierher?» fragte der Doktor,
seinerseits erschreckend.

«Ich und Mr. Giles haben heute morgen nach ihnen geschickt,» antwortete
Brittles, «und ich wundere mich nur, daß sie so spät kommen.»

«Ah, Sie schickten nach ihnen! Ei, so wollt' ich, daß dieser und jener
Sie holte! Ihr seid hier doch lauter verwünschte Dummköpfe!» sagte der
Doktor im Hinauseilen.




31. Kapitel.

    Eine kritische Situation.


«Wer ist hier?» fragte Brittles, indem er die Haustür ein wenig öffnete
und die Kerze mit der Hand beschattend, hinausschaute.

«Öffnen Sie die Tür», entgegnete ein Mann von draußen. «Es sind die
Polizeibeamten aus Bow-Street, nach denen heut' geschickt worden ist.»

Durch diese Auskunft völlig beruhigt, öffnete Brittles die Tür in ihrer
vollen Breite und stand einem stattlichen Manne in einem großen Mantel
gegenüber, der sofort ohne weiteres eintrat und sich die Stiefel so
ruhig auf der Matte reinigte, als gehöre er ins Haus.

«Schicken Sie sofort jemand, der meinem Kollegen die Sorge für Pferd
und Wagen abnimmt. Haben Sie nicht eine Remise hier, daß wir den Wagen
kurze Zeit unterstellen können?»

Als Brittles eine bejahende Antwort gab und auf das Gebäude deutete,
schritt der stattliche Mann zur Gartenpforte zurück und half seinem
Kollegen beim Aussteigen aus dem Gig, wobei ihnen Brittles mit dem
Ausdruck hoher Bewunderung leuchtete. Hierauf kehrten beide Beamte nach
dem Hause zurück und legten, ins Besuchszimmer geleitet, ohne weiteres
Überrock und Hut ab. Der erste, der geklopft hatte, war ein starker
Mann von Mittelgröße, etwa fünfzig Jahre alt, und hatte glänzendes,
ziemlich kurz geschnittenes Haar, ein rundes Gesicht und scharfe Augen.
Der andere war ein Rotkopf und hager, trug Stulpenstiefel und hatte ein
abstoßendes Gesicht und eine aufgeworfene, widerwärtige Nase.

«Melden Sie Ihrer Herrschaft, daß Blathers und Duff hier wären», sagte
der stattlichere von beiden, sein Haar niederstreichend und ein Paar
Handfesseln auf den Tisch legend. «Ah! guten Abend, Sir. Kann ich ein
Wörtchen allein mit Ihnen reden?»

Diese Anrede war an Mr. Losberne gerichtet, der eben mit den beiden
Damen eintrat und Brittles einen Wink gab, hinauszugehen. «Dies ist die
Dame des Hauses», sagte Losberne mit einer Handbewegung auf Mrs. Maylie
zu.

Mr. Blathers machte eine Verbeugung. Auf die Aufforderung, Platz
zu nehmen, stellte er seinen Hut auf den Fußboden, setzte sich und
veranlaßte Duff, das gleiche zu tun. Der letztere, der sich weniger in
guter Gesellschaft bewegt zu haben schien oder sich doch jedenfalls
nicht mit großer Leichtigkeit darin bewegte, nahm erst nach manchem
umständlichen Kratzfuße Platz und legte dann sofort den Knauf seines
Handstockes an den Mund.

«Lassen Sie uns nun aber sogleich auf den hier verübten Einbruch
kommen, Sir», sagte Blathers. «Wie verhält es sich mit der Sache?»

Losberne wünschte Zeit zu gewinnen und berichtete der Länge nach und
mit großer Weitschweifigkeit. Die Herren Blathers und Duff hörten mit
äußerst weisen Mienen zu und blinzelten einander dann und wann sehr
pfiffig zu.

«Ich kann über die Sache natürlich nicht eher etwas Gewisses sagen,»
bemerkte Blathers, als Losberne mit seinem Bericht zu Ende war, «als
bis ich die Stelle in Augenschein genommen habe, wo der Einbruch
versucht worden ist; jedoch meine Meinung ist rund heraus die -- denn
ich stehe, selbst auf die Gefahr, zu irren, nicht an, so weit zu gehen
-- daß er von keinem Kaffer verübt ist -- was sagst du, Duff?»

Duff war derselben Meinung.

«Sie wollen sagen,» versetzte Losberne lächelnd, «der Einbruch sei von
keinem Landmanne, von keinem Nicht-Londoner verübt?»

«Ganz recht, Sir. Wissen Sie noch etwas über das Verbrechen zu sagen?»

Losberne verneinte.

«Was ist denn das aber mit dem Knaben, von dem die Dienerschaft im
Hause spricht?»

«O ganz und gar nichts», erwiderte der Doktor. «Der Haushofmeister
hatte es sich in seiner Bestürzung in den Kopf gesetzt, der Knabe wäre
bei dem Einbruche, der Himmel weiß wie, beteiligt gewesen -- 's ist
aber durchaus nichts als Torheit und alberne Einbildung gewesen.»

«Das heißt die Sache gar zu sehr auf die leichte Achsel nehmen»,
bemerkte Duff.

«Du hast ganz recht, Duff», sagte Blathers mit bekräftigendem
Kopfnicken und mit den Handfesseln spielend, als wenn sie ein Paar
Kastagnetten gewesen wären. «Wer ist der Knabe? Welche Auskunft gibt er
über sich? Woher kam er? Er wird doch nicht aus den Wolken gefallen
sein, Sir?»

«Natürlich, nein», sagte Losberne, den Damen einen unruhigen Blick
zuwerfend. «Mir ist indessen sein ganzer Lebenslauf bekannt, und --
doch wir können nachher darüber sprechen. Wollen Sie nicht vor allen
Dingen die Stelle sehen, wo die Diebe einzubrechen versuchten?»

«Allerdings,» erwiderte Blathers. «Wir nehmen zuerst die Stelle in
Augenschein und verhören sodann die Dienerschaft. Das pflegt der
gewöhnliche Gang des Geschäfts zu sein.»

Es wurde Licht gebracht, und die Herren Blathers und Duff, in
Begleitung des Konstablers des Ortes, Brittles', Giles' und, mit einem
Worte, sämtlicher sonstiger Hausbewohner, begaben sich in das kleine
Gemach am Ende des Hausflurs und sahen aus dem Fenster, gingen darauf
hinaus und sahen in das Fenster hinein, besichtigten den Fensterladen,
spürten den Fußtritten nach beim Scheine einer Laterne und durchstachen
die Büsche vermittels einer Heugabel. Nachdem dies alles geschehen war
und alle das Vorgehen der Beamten mit atemloser Teilnahme verfolgt
hatten, gingen Blathers und Duff wieder hinein und vernahmen Giles und
Brittles über ihren Anteil an den Begebenheiten der Schreckensnacht;
beide Diener erzählten sechsmal statt einmal und widersprachen einander
beim ersten nur in einem einzigen wichtigen Punkte und beim letzten nur
in einem Dutzend wesentlicher Aussagen. Nach Beendigung des Verhörs
wurden Giles und Brittles entlassen, und die Herren Blathers und Duff
hielten eine lange Beratung ab, im Vergleich zu der in Beziehung auf
Heimlichkeit und Feierlichkeit eine Konsultation berühmter Doktoren
über den schwierigsten Krankheitsfall bloßes Kinderspiel gewesen wäre.

Losberne ging unterdessen im anstoßenden Zimmer sehr unruhig auf und
ab, und Mrs. Maylie und Rose schauten ihm mit noch größerer Unruhe zu.

«Auf mein Wort,» sagte er, plötzlich stillstehend, «ich weiß kaum, was
hier zu tun ist.»

«Wenn den beiden Männern», versetzte Rose, «die Geschichte des
unglücklichen Knaben erzählt würde, wie sie ist, es wäre sicher genug,
ihn in ihren Augen von Schuld zu entlasten.»

«Das muß ich bezweifeln, meine werte junge Dame», wandte der Doktor
kopfschüttelnd ein. «Ich glaube nicht, daß es sie oder auch die höheren
Polizei- oder Justizbeamten befriedigen würde. Sie würden sagen, er sei
jedenfalls ein fortgelaufener Kirchspielknabe und Lehrling. Nach rein
weltlich-verständigen Erwägungen und Wahrscheinlichkeiten beurteilt,
unterliegt seine Geschichte großen Zweifeln.»

«Sie schenken ihr doch Glauben?» fiel Rose hastig ein.

«Ich schenke ihr Glauben, so befremdlich sie lautet, und bin vielleicht
ein großer Tor, weil ich es tue,» versetzte der Doktor; «allein
nichtsdestoweniger halte ich sie keineswegs für eine solche, die einen
erfahrenen Polizeibeamten zufriedenstellen würde.»

«Warum aber nicht?» fragte Rose.

«Meine schöne Inquirentin,» erwiderte Losberne, «weil in ihr, wenn
man sie mit den Augen jener Herren betrachtet, so viele böse Umstände
vorkommen. Der Knabe kann nur beweisen, was übel, und nichts von dem,
was gut aussieht. Die verwünschten Spürhunde werden nach dem Warum
und Weshalb fragen und nichts als wahr gelten lassen, was ihnen nicht
vollständig bewiesen wird. Er sagt selbst, daß er sich eine Zeitlang in
der Gesellschaft von Diebesgelichter befunden, eines Taschendiebstahls
angeklagt vor einem Polizeiamte gestanden hat, und aus dem Hause
des bestohlenen Herrn gewaltsam entführt ist, er kann selbst nicht
angeben, hat nicht einmal eine Vermutung, wohin. Er wird von Männern
nach Chertsey hergebracht, die ganz vernarrt in ihn zu sein scheinen
und ihn durch ein Fenster stecken, um ein Haus zu plündern; und gerade
in dem Augenblicke, wo er die Bewohner wecken und tun will, was
seine Unschuld ins Licht setzen würde, verrennt ihm der verwünschte
Haushofmeister den Weg und schießt ihn in den Arm, gleichsam recht
absichtlich, um ihn daran zu hindern, etwas zu tun, das ihm nützen
könnte. Leuchtet Ihnen das alles nicht ein?»

«Natürlich leuchtet es mir ein», erwiderte Rose, den Eifer des Doktors
belächelnd; «allein ich sehe nur noch immer nichts darin, wodurch die
Schuld des armen Kindes erwiesen würde.»

«Nicht -- ei!» rief Losberne aus. «O, über die hellen, scharfen
Äugelein der Damen, womit sie, sei es zum Guten oder Bösen, immer nur
die eine Seite an einer Sache oder Frage sehen, und zwar stets die, die
sich ihnen eben zuerst dargeboten hat!»

Nachdem er seinem Herzen Luft dadurch gemacht, daß er Miß Rose diesen
Erfahrungssatz zu Gemüt geführt, steckte er die Hände in die Taschen
und fing wieder an, mit noch rascheren Schritten als zuvor im Zimmer
auf und ab zu gehen. «Je mehr ich darüber nachdenke,» fuhr er fort,
«desto zahlreichere und größere Schwierigkeiten sehe ich voraus, den
beiden Leuten die Geschichte des Knaben glaubhaft zu machen. Ich bin
überzeugt, daß sie ihm schlechterdings keinen Glauben schenken werden,
und selbst wenn sie ihm am Ende nichts anhaben können, so werden doch
ihre Zweifel und der Verdacht, den diese wieder auf ihn werfen müssen,
von sehr wesentlichem Nachteile für den wohlwollenden Plan sein, ihn
aus dem Elende zu retten.»

«O bester Doktor, was ist da zu tun?» rief Rose aus. «Du lieber Himmel,
daß Giles auch den unseligen Einfall hat haben müssen, nach der Polizei
zu schicken!»

«Ich wüßte nicht, was ich darum gäbe, wenn es nicht geschehen wäre»,
fiel Mrs. Maylie ein.

«Ich weiß nur eins,» sagte der Doktor, sich mit einer Art Ruhe
der Verzweiflung hinsetzend, «daß ich die Kerle mit göttlicher
Unverschämtheit aus dem Hause zu bringen suchen muß. Der Zweck ist ein
guter, und darin liegt die Entschuldigung. Bei dem Knaben zeigen sich
starke Fiebersymptome, und er befindet sich in einem Zustande, daß er
für jetzt nicht mehr befragt werden darf; das ist ein Trost. Wir müssen
seine Lage so gut wie möglich zu benutzen suchen, und wenn es nicht
glücken will, so ist es nicht unsere Schuld. Herein!»

Blathers und Duff erschienen, und der erstere sprach sogleich ein
Urteil über den Einbruch in einem Kauderwelsch aus, das weder Losberne
noch die Damen verstanden. Um eine Erklärung gebeten, sagte er, dem
Doktor einen verächtlichen Blick zuwerfend und sich mitleidig zu den
Damen wendend, er meine, daß die Dienerschaft bei dem beabsichtigten
Raube nicht im Einverständnisse gewesen sei.

«Wir haben auch durchaus keinen Verdacht gegen sie gehabt», bemerkte
Mrs. Maylie.

«Mag wohl sein, Ma'am», entgegnete Blathers; «sie konnte aber auch Hand
im Spiele gehabt haben.»

«Und eben weil kein Verdacht sie traf,» fiel Duff ein, «mußte um so
mehr danach geforscht werden.»

«Wir haben gefunden, daß der Einbruch Londoner Werk ist», fuhr Blathers
fort; «die Kerle haben meisterhaft gearbeitet.»

«In Wahrheit sehr wackere Arbeit», bemerkte Duff leise.

«Der Einbrecher sind zwei gewesen,» berichtete Blathers weiter, «und
sie haben einen Knaben bei sich gehabt, was aus der Größe des Fensters
klar ist. Mehr läßt sich für jetzt nicht sagen. Zeigen Sie uns doch den
Burschen, den Sie im Hause haben.»

«Die Herren nehmen aber wohl erst ein wenig zu trinken an, Mrs.
Maylie», sagte der Doktor mit erheiterten Mienen, als wenn ihm ein
neuer Gedanke aufgegangen wäre.

«Gewiß», fiel Rose eifrig ein. «Es steht Ihnen sogleich alles zu
Diensten, wenn Sie befehlen.»

«Besten Dank, Miß», sagte Blathers, mit dem Rockärmel über den Mund
fahrend. «So ein Verhör ist trockene Arbeit. Was Sie eben zur Hand
haben, Miß; machen Sie sich unsertwegen keine Ungelegenheiten.»

«Was belieben Sie?» fragte der Doktor, der jungen Dame nach dem
Eckschrank folgend.

«Wenn's Ihnen gleichviel ist, 'nen Tropfen Branntwein, Sir», erwiderte
Blathers. «Wir hatten 'ne kalte Fahrt von London her, und der
Branntwein läuft einem so warm übers Herz.»

Er richtete die letzteren Worte an Mrs. Maylie, und der Doktor
schlüpfte unterdes aus dem Zimmer.

«Ah, meine Damen,» fuhr Blathers, das ihm gereichte Glas vor das Auge
emporhaltend, fort, «ich habe in meinem Leben die schwere Menge solcher
Geschichten erlebt.»

«Zum Beispiel den Einbruch in Edmonton, Blathers», fiel Duff ein.

«Ja, ja», sagte Blathers; «der war diesem allerdings ähnlich genug. Er
wurde von dem Conkey Chickweed begangen.»

«Das hast du immer behauptet», entgegnete Duff; «aber ich sage dir,
die Familie Pet hat ihn verübt, und Conkey hat nicht mehr die Hand im
Spiele dabei gehabt als ich.»

«Ei was,» sagte Blathers, «ich weiß es besser. Entsinnst du dich noch,
wie sich Conkey sein Geld stehlen ließ? Es war 'ne Geschichte, noch
merkwürdiger, als sie in 'nem Buche vorkommen kann.»

«Erzählen Sie doch», nahm Rose das Wort, um die unwillkommenen Gäste
bei guter Laune zu erhalten.

«Es war 'ne Spitzbüberei, worauf so leicht niemand verfallen sein
würde, Miß», begann Blathers. «Nämlich der Conkey Chickweed --»

«Conkey bedeutet soviel als Emmesgatsche[AN], Ma'am», bemerkte Duff.

  [AN] Verräter, Angeber.

«Das wird die Dame ja wohl wissen», bemerkte Blathers. «Unterbrich
mich doch nicht immer. Also, Miß, der Conkey Chickweed hatte ein
Wirtshaus oberhalb Battle-Bridge und 'nen Raum, den viele junge
Lords besuchten, um den Hahnenkämpfen, Dachshetzen und dergleichen
zuzuschauen, was man nirgends besser sehen konnte. Er gehörte zu der
Zeit noch nicht zur Kabrusche[AO], und einst wurden ihm mitten in der
Nacht dreihundertsiebenundzwanzig Guineen aus seiner Schlafkammer von
'nem großen Manne mit 'nem schwarzen Pflaster über dem einen Auge
gestohlen, der sich unter dem Bett versteckt gehabt hatte und mit dem
Gelde aus dem Fenster sprang. Er war dabei flink genug; Conkey aber
war auch geschwind; das Geräusch hatte ihn aufgeweckt; er sprang aus
dem Bette, schoß hinter dem Diebe drein und machte die Nachbarn wach.
Sie erhoben sogleich ein allgemeines Hallo und fanden, daß Conkey den
Dieb getroffen haben mußte, denn sie entdeckten und verfolgten auf
einer ganzen Strecke Blutspuren, die sich indes endlich verloren. Das
Geld war fort, und Chickweed machte Bankerott. Er ging ein paar Tage
ganz außer sich umher, zerraufte sich das Haar und erregte so sehr das
allgemeine Mitleid, daß ihm von allen Seiten milde Gaben zugeschickt,
Subskriptionen für ihn eröffnet wurden usw. Eines Tages kam er in das
Polizeibureau hereingestürzt und hatte eine geheime Unterredung mit dem
Friedensrichter, der darauf Jem Spyers (Jem war einer der tätigsten
Geheimpolizisten) beorderte, Chickweed bei Gefangennehmung des Diebes
Beistand zu leisten. >Spyers<, sagte Chickweed, >ich habe ihn gestern
morgen vor meinem Hause vorbeigehen sehen.< -- >Warum haben Sie ihn
nicht sogleich angehalten?< fragte Spyers. >Ich war so bestürzt, daß
Sie mir den Hirnschädel mit 'nem Zahnstocher hätten entzweischlagen
können,< antwortete der arme Mensch; >wir werden ihn aber gewiß
attrapieren, denn heut' abend zwischen zehn und elf Uhr kommt er
wieder vorüber.< Spyers ging sogleich mit ihm und pflanzte sich an
ein Wirtshausfenster hinter den Vorhang. Er rauchte in guter Ruh',
aber mit dem Hut auf dem Kopfe, seine Pfeife, als Chickweed plötzlich
anfing zu schreien: >Da ist er! Haltet den Dieb! Mordjo, mordjo!< Jem
Spyers stürzte hinaus und sah Chickweed im vollen Laufe hinter dem
Diebe herrennen. Er fing auch an zu laufen, was hast du, was kannst du,
geriet endlich ins Gedränge und fand Chickweed darin wieder, allein der
Dieb war entkommen, was merkwürdig genug war. Am anderen Morgen war
Spyers abermals auf seinem Posten, sah sich die Augen nach 'nem großen
Manne mit 'nem schwarzen Pflaster müde, so daß er sie endlich mal
wegwenden und ruhen lassen mußte, und im selbigen Augenblick, als er's
tat, fing Chickweed wiederum an zu schreien. Jem stürzt hinaus und ihm
nach, sie laufen zweimal so weit wie am vorigen Tage, und endlich ist
der Dieb wiederum zum Geier. Und so ging's noch mehrere Male, so daß
die Nachbarn sagten, der Teufel selbst hätte Chickweed bestohlen und
spielte ihm hinterher noch schlechte Streiche; andere aber sagten, der
unglückliche Chickweed wäre vor Kummer verrückt geworden.»

  [AO] Gaunergenossenschaft.

«Was sagte denn Jem Spyers?» fragte der Doktor, der wieder in das
Zimmer zurückgekehrt war.

«Jem Spyers,» erwiderte der Erzähler, «sagte 'ne lange Zeit gar nichts
und horchte auf alles, ohne daß man's ihm ansah, zum Zeichen, daß
er sich auf sein Geschäft verstand. Eines Morgens aber trat er zu
Chickweed und sagte: >Guter Freund, ich hab's jetzt heraus, wer den
Diebstahl begangen hat.< -- >Wirklich!< rief Chickweed aus; >o mein
bester Spyers, machen Sie nur, daß ich mich an dem Halunken rächen
kann, so werd' ich dermaleinst zufrieden sterben. Bester Spyers, wie
heißt der Bösewicht?< -- >Guter Freund,< antwortete Spyers, ihm eine
Prise anbietend, >lassen Sie die Narretei! Sie haben es selbst getan.<
Und so war's auch, Chickweed hatte sich dadurch ein anständiges Stück
Geld gemacht, und es würde auch niemand dahintergekommen sein, wenn er
nicht so übereifrig gewesen wäre, den Verdacht von sich fernzuhalten.»

«Ein seltsamer Fall», bemerkte der Doktor. «Wenn es Ihnen aber beliebt,
so können Sie jetzt hinaufgehen.»

Die beiden Konstabler begaben sich mit Losberne in Olivers Zimmer.
Giles leuchtete ihnen. Der kleine Patient hatte geschlummert und sah
kränker und fieberischer aus als am Tage. Der Doktor stützte ihn, so
daß er sich eine kurze Weile emporrichten konnte, und er starrte umher,
ohne zu wissen, was mit ihm vorging, oder sich zu erinnern, wo er sich
befand oder was mit ihm vorgegangen war.

«Dies ist der Knabe,» sagte Losberne leise, aber dessenungeachtet mit
großer Lebhaftigkeit, «der in einem Garten hier in der Nähe bei einer
kleinen Übertretung, wie sie bei Kindern häufig vorkommt, durch einen
Selbstschuß verwundet ist, in Mrs. Maylies Hause Beistand gesucht, und
den der scharfblickende Herr da mit dem Lichte in der Hand sogleich
festgehalten und dermaßen mißhandelt hat, daß das Leben des Patienten,
was ich ärztlich bescheinigen kann, beträchtlich gefährdet worden ist.»

Blathers und Duff hefteten die Blicke auf den solchermaßen ihrer
Beachtung empfohlenen Giles, dessen Mienen das spaßhafteste Gemisch von
Furcht und Verwirrung ausdrückten.

«Sie werden nicht leugnen wollen?» fügte Losberne, Oliver wieder
niederlegend, hinzu.

«Es geschah alles zum -- zum Besten, Sir!» antwortete Giles. «Ich hielt
ihn für den Knaben; hätte mich sonst sicher nicht mit ihm befaßt. Ich
bin wahrlich kein Unmensch, Sir.»

«Für was für 'nen Knaben hielten Sie ihn?» fragte Blathers.

«Für den Gehilfen der Einbrecher», erwiderte Giles. «Sie -- sie hatten
einen Knaben bei sich.»

«Halten Sie ihn jetzt noch für den Knaben?»

«Kann's wirklich nicht sagen -- könnt's nicht beschwören, daß er es
ist.»

«Was glauben Sie aber?»

«Ich weiß wirklich nicht, was ich glauben soll. Ich glaube nicht, daß
es der Knabe ist; ich bin so gut wie gewiß, daß er es nicht ist, Sie
wissen, daß er es nicht sein kann.»

«Hat der Mann getrunken, Sir?» fragte Blathers den Doktor.

Losberne hatte unterdes Olivers Puls gefühlt, stand auf und bemerkte,
die Herren möchten, wenn sie Zweifel hegten, im anstoßenden Zimmer
Brittles befragen. Man begab sich in das anstoßende Zimmer, und
Brittles wurde gerufen und verwickelte sich, wie Mr. Giles, in ein
solches Irrsal neuer Widersprüche und Unmöglichkeiten, daß durchaus
nichts klar wurde als seine eigene Unklarheit, und daß nur einige
seiner Aussagen einiges Licht gaben: er würde den Knaben nicht
wiedererkennen, hätte Oliver nur für denselben gehalten, weil Giles
gesagt, daß er es wäre, und Giles hätte noch vor fünf Minuten in der
Küche erklärt, daß er zu voreilig gewesen zu sein fürchte.

Unter anderen scharfsinnigen Fragen wurde auch die aufgeworfen, ob
Mr. Giles wirklich jemand getroffen habe, und als sein zweites Pistol
untersucht wurde, fand sich, daß es nur mit Pulver geladen war, --
eine Entdeckung, welche großen Eindruck auf alle machte, den Doktor
ausgenommen, der zehn Minuten zuvor die Kugel herausgezogen hatte.
Den größten Eindruck machte sie aber auf Mr. Giles selbst, der in der
schrecklichsten Angst geschwebt hatte, ein unglückliches Kind verwundet
zu haben, und nunmehr nach Kräften die Vermutung begünstigte, daß auch
das erste Pistol nur mit Pulver geladen gewesen sei. Endlich entfernten
sich Blathers und Duff, ohne sich um Oliver viel zu kümmern, den
Konstabler aus Chertsey zurücklassend und unter dem Versprechen, am
anderen Morgen wiederzukommen.

Am anderen Morgen verbreitete sich in dem Städtchen, in welchem sie
übernachtet, das Gerücht, daß zwei Männer und ein Knabe in der Nacht
unter verdächtigen Umständen angehalten und nach Kingston gebracht
wären, wohin sich demgemäß Blathers und Duff begaben. Die verdächtigen
Umstände schrumpften indes bei genauerer Nachforschung zu dem einen
Umstande zusammen, daß die Delinquenten in einem Heuschober geschlafen
hatten, was, obwohl ein großes Verbrechen, doch nur mit Gefängnis
bestraft werden kann und in den gnadenvollen Augen des englischen, mit
gemeinsamer Liebe alle Untertanen umfassenden Gesetzes, in Ermangelung
aller sonstigen Indizien, nicht als genügender Beweis gilt, daß der
oder die Schläfer gewaltsamen Einbruch begangen haben und deshalb der
Todesstrafe verfallen sind. Blathers und Duff kehrten daher gerade so
klug zurück, wie sie hingereist waren.

Kurz, nach mehrfachen Verhandlungen ließ sich der nächstwohnende
Friedensrichter leicht bewegen, Mrs. Maylies und Mr. Losbernes
Bürgschaft für Olivers Erscheinen vor Gericht anzunehmen, falls
er zitiert werden sollte, und Blathers und Duff gingen, nachdem
sie durch ein paar Guineen belohnt waren, mit geteilten Meinungen
nach London zurück, indem der letztere, nach reiflicher Überlegung
aller betreffenden Umstände, zu der Annahme hinneigte, daß der
Einbruchsversuch von der Familie Pet ausgegangen sei, wogegen der
erstere ebenso sehr geneigt war, das ganze Verdienst der Tat dem großen
Conkey Chickweed zuzuschreiben.

Mit Oliver besserte es sich unter der vereinten sorgfältigen Behandlung
und Pflege Mrs. Maylies, Roses und des gutherzigen Doktors. Wenn
glühende Bitten, aus Herzen von Dankbarkeit überfließend, im Himmel
erhört werden -- und was sind Gebete, wenn der Himmel sie nicht erhört?
--, so vernahm er die Segnungen, die das verwaiste Kind auf seine
Wohltäter herabflehte, die dadurch mit Friede und Freude in ihrem
Innern belohnt wurden.




32. Kapitel.

    Von dem glücklichen Leben, das Oliver bei seinen gütigen
    Gönnerinnen zu führen anfing.


Oliver litt nicht wenig. Zu den Schmerzen der Schußwunde kam noch
ein heftiges Fieber, die Folge der Kälte und Nässe, der er nach
seiner Verwundung ausgesetzt gewesen war. Er lag mehrere Wochen fest
zu Bett, fing indes allmählich an zu genesen und konnte bald unter
Tränen mit wenigen Worten ausdrücken, wie tief er die Güte der beiden
freundlichen, liebevollen Damen empfände, und wie sehr er wünschte
und hoffte, wenn er wiederhergestellt wäre, imstande zu sein, ihnen
Beweise seiner Dankbarkeit zu geben, etwas zu tun, und wenn es noch so
wenig wäre, ihnen die Liebe zu zeigen, die sein Herz erfüllte, ihnen
die Überzeugung zu verschaffen, daß sie ihre Güte an keinen Unwürdigen
verschwendeten, sondern daß der arme, verlassene Knabe, den sie vom
Elende oder Tode errettet, den glühenden Wunsch hege, ihnen nach all
seinen Kräften und mit tausend Freuden zu dienen.

«Armes Kind!» sagte Rose, als er eines Tages mit bleichen Lippen Worte
des Dankes zu stammeln versuchte. «Du sollst viele Gelegenheiten
erhalten, uns zu dienen, wenn du willst. Wir gehen auf das Land, und
meine Tante beabsichtigt, dich mitzunehmen. Die ländliche Ruhe, die
reine Luft und die Freuden und Schönheiten des Frühlings werden bald
deine gänzliche Genesung bewirken, und wir wollen dir hundert kleine
Geschäfte auftragen, sobald du der Mühe gewachsen bist.»

«Der Mühe!» sagte Oliver. «Ach, wenn ich nur für Sie arbeiten -- Ihnen
nur Freude machen könnte, dadurch, daß ich Ihre Blumen begösse, Ihre
Vögel fütterte, den ganzen Tag hin und wieder für Sie liefe, was würde
ich darum geben!»

«Du sollst gar nichts darum geben,» versetzte Rose lächelnd, «denn wie
ich es dir schon gesagt habe, wir denken dich auf die vielfachste Weise
zu beschäftigen, und du wirst mir die größte Freude bereiten, wenn du
nur halb so viel tust, wie du jetzt versprichst.»

«Ihnen Freude bereiten -- o wie gütig Sie sind!» rief Oliver aus.

«Du wirst mir mehr Freude bereiten, als ich es dir sagen kann»,
versetzte die junge Dame. «Es gewährt mir schon unsägliches Vergnügen,
zu denken, daß meine liebe, gute Tante ein Werkzeug in den Händen der
Vorsehung gewesen ist, einen Knaben aus einer so entsetzlichen Lage
zu erretten, wie du sie uns beschrieben hast; allein zu erfahren,
daß ihr kleiner Schützling dankbar und liebevoll gegen sie für ihre
Wohltätigkeit und ihr Mitleid ist, wird mich weit glücklicher machen,
als du es dir vorstellen kannst. Verstehst du mich, Oliver?» fragte
sie, Olivers nachdenkliches Gesicht betrachtend.

«O ja, ja, ich verstehe Sie wohl; aber ich meinte nur, daß ich jetzt
undankbar wäre.»

«Gegen wen denn?»

«Gegen den gütigen Herrn und die gute alte Frau, denen ich so große
Wohltaten verdanke, die sich meiner so liebevoll annahmen. Gewiß, sie
würden sich freuen, wenn sie es wüßten, wie gut ich es hier habe.»

«Das glaube ich auch, und Mr. Losberne hat schon versprochen, dich
mitzunehmen zu ihnen, sobald es deine Kräfte erlauben würden.»

«Hat er das versprochen? Oh, ich weiß nicht, was ich vor Freude tun
werde, wenn ich sie einmal wiedersehe!»

Oliver war nach einiger Zeit hinlänglich wiederhergestellt, um stark
genug zu einer Ausfahrt zu sein, und fuhr eines Morgens mit Mr.
Losberne in Mrs. Maylies Wagen ab. Als sie an die Brücke von Chertsey
kamen, erblaßte Oliver plötzlich und stieß einen lauten Ausruf der
Überraschung und Bestürzung aus.

«Was gibt es?» rief der Doktor mit seiner gewöhnlichen Lebhaftigkeit.
«Siehst du etwas -- hörst du etwas -- hast du Schmerz -- was gibt's?»

«Da, Sir!» sagte Oliver, aus dem Wagenfenster zeigend. «Da, jenes Haus!»

«Was ist mit dem Hause? Halt, Kutscher! -- He -- was willst du sagen?»

«Die Diebe -- in das Haus schleppten sie mich», flüsterte Oliver.

«Ist es möglich! Halt, halt, Kutscher!» rief der Doktor, sprang aus dem
Wagen, noch ehe derselbe hielt, lief nach dem verödet aussehenden Hause
und fing an wie toll gegen die Tür zu hämmern.

«Zum Teufel, was soll das?» sagte ein kleiner, häßlicher, buckliger
Mann, der die Tür so plötzlich öffnete, daß Losberne fast in das Haus
hineingefallen wäre.

«Was das soll?» rief der Doktor, ihn ohne Umstände beim Kragen fassend.
«Sehr viel soll's. Es handelt sich um Diebstahl und Einbruch.»

«Und es wird sich auch sogleich um Mord handeln,» erwiderte der
Bucklige kaltblütig, «wenn Sie nicht sogleich von mir ablassen. Hören
Sie?»

«Ich höre sehr wohl», sagte der Doktor, ihn kräftig schüttelnd. «Wo ist
-- wie heißt der verwünschte Halunke gleich -- ja, Sikes -- Spitzbube,
wo ist Sikes?»

Der Bucklige starrte ihn erstaunt und wütend an, entschlüpfte ihm,
stieß eine Flut der schrecklichsten Verwünschungen aus und ging in
das Haus zurück. Bevor er jedoch die Tür wieder verschließen konnte,
stürmte der Doktor in das nächste Zimmer hinein und blickte forschend
umher, allein von allem, was er sah, wollte nichts mit Olivers
Beschreibung zusammenstimmen.

«Was soll das bedeuten, daß Sie auf solche Weise in mein Haus
eindringen?» fragte nach einigen Augenblicken der Bucklige, der ihn
scharf beobachtet hatte. «Wollen Sie mich bestehlen oder ermorden?»

«Hast du jemals einen Mann in solcher Absicht aus einer Equipage
aussteigen sehen, du lächerliche, alte Mißgeburt?» lautete des
reizbaren Doktors Gegenfrage.

«Was wollen Sie denn aber sonst?» fuhr ihn der Bucklige an. «Packen Sie
sich augenblicklich aus meinem Hause, oder es wird Sie reuen.»

«Ich werde gehen, sobald es mir beliebt,» sagte Losberne, in das andere
Zimmer hineinblickend, das gleichfalls keine Ähnlichkeit mit dem
von Oliver beschriebenen hatte, «und will dir schon noch hinter die
Schliche kommen!»

«So!» höhnte der Krüppel. «Wenn Sie mich suchen, ich bin hier zu
finden. Ich habe hier nicht als ein Verrückter und ganz allein seit
fünfundzwanzig Jahren gewohnt, um mich von Ihnen hudeln zu lassen. Sie
sollen mir dafür büßen, sollen mir dafür büßen!»

Der mißgestaltete kleine Dämon fing darauf an, auf das schrecklichste
und ungebärdigste zu schreien und zu toben, der Doktor murmelte vor
sich hin: «Dumme Geschichte! Der Knabe muß sich geirrt haben», warf
dem Buckligen ein Stück Geld zu und kehrte zu dem Wagen zurück. Der
Bucklige folgte ihm unter beständigen Schimpfreden und Verwünschungen,
sah, während Losberne dem Kutscher ein paar Worte sagte, in den Wagen
hinein und warf Oliver einen so grimmigen, stechenden, rachsüchtigen
und giftigen Blick zu, daß ihn der kleine Rekonvaleszent monatelang
wachend oder schlafend nicht wieder vergessen konnte. Losberne stieg
ein, und sie fuhren ab, hörten aber den Buckligen noch lange schreien
und toben, der sich vor Wut schäumend das Haar zerraufte, mit den Füßen
stampfte und ganz außer sich zu sein schien.

«Ich bin ein Esel!» sagte der Doktor nach einem langen Stillschweigen.
«Hast du das schon gewußt, Oliver?»

«Nein, Sir.»

«Dann vergiß es ein anderes Mal nicht. -- Selbst wenn es das Haus
war,» fuhr er nach einer abermaligen Pause fort, «und die Diebe darin
gewesen wären -- was hätt' ich als einzelner tun können? Und hätt'
ich Beistand gehabt, so wäre auch nichts weiter dabei herausgekommen,
als daß meine Voreiligkeit und die Weise kund geworden, wie ich den
unangenehmen Handel zu vertuschen gesucht. Es wäre mir freilich gerade
recht geschehen, und ich würde nicht dümmer danach geworden sein, denn
ich bringe mich selbst in eine Patsche nach der andern, indem ich immer
bloß nach den fatalen Eindrücken des Augenblicks handle.»

Der treffliche Doktor hatte in seinem ganzen Leben nur nach ihnen
gehandelt, und es lag kein geringes Lob der in ihm vorherrschenden oder
ihn bestimmenden Eindrücke in dem Umstande, daß er, weit entfernt,
jemals in ernstliche Unannehmlichkeiten durch sie geraten zu sein, bei
allen, die ihn kannten, die wärmste und größte Hochachtung genoß.
Muß die Wahrheit gesagt sein, so war er ein paar Minuten übler Laune,
sich in der Hoffnung getäuscht zu sehen, sogleich bei der ersten sich
darbietenden Gelegenheit Zeugnisse für die Wahrheit der Erzählung
Olivers zu erhalten. Sein Gleichmut war jedoch bald wiederhergestellt,
und da die Antworten des Knaben auf seine erneuerten Fragen klar
und zusammenhängend waren und blieben und mit derselben offenbaren
Aufrichtigkeit wie früher gegeben wurden, so nahm er sich vor, ihnen
von nun an vollkommenen Glauben zu schenken.

Da Oliver die Straße zu nennen wußte, in welcher Mr. Brownlow wohnte,
so waren keine Kreuz- und Querfragen erforderlich, und als sie
hineinfuhren, klopfte des Knaben Herz so heftig, daß er kaum zu atmen
imstande war. Losberne forderte ihn auf, das Haus zu bezeichnen.

«Das da!» rief Oliver, eifrig aus dem Fenster zeigend. «Das weiße Haus!
Oh, lassen Sie rasch fahren, recht rasch. Es ist mir, als wenn ich
sterben müßte, eh' ich hinkomme; ich kann mir vor Zittern nicht helfen!»

«Nur Geduld, mein lieber Kleiner», sagte Losberne, ihn auf die Schulter
klopfend. «Du wirst deine Freunde sogleich sehen, und sie werden sich
freuen, dich gesund und wohlbehalten wiederzufinden.»

«Oh, das hoff' ich auch», versetzte Oliver. «Sie waren so gütig, so
sehr, sehr gütig gegen mich, Sir.»

Der Wagen hielt, und Oliver blickte mit Tränen der freudigsten
Erwartung nach den Fenstern hinauf. Doch ach! Das weiße Haus war
unbewohnt; ein Anschlag verkündigte, daß es zu vermieten sei. Losberne
stieg aus, zog den Knaben mit sich fort, klopfte an die nächste Tür
und fragte die öffnende Magd, ob sie wisse, wohin sich Mr. Brownlow
gewendet, der nebenan gewohnt habe. Sie wußte es nicht, lief hinauf, um
sich zu erkundigen, kehrte zurück und brachte die Nachricht, er habe
sein Haus und seine Mobilien verkauft und sei vor sechs Wochen nach
Westindien gegangen. Oliver schlug die Hände zusammen und wäre bald zu
Boden gesunken.

«Hat er auch seine Haushälterin mitgenommen?» fragte Losberne nach
einem kurzen Stillschweigen.

«Ja, Sir; der alte Herr, die Haushälterin und ein Freund von ihm sind
miteinander abgereist.»

«Wir kehren sogleich nach Hause zurück», rief der Doktor dem Kutscher
zu; «und fahren Sie rasch, daß wir sobald wie möglich aus dem
verwünschten London wieder hinauskommen.»

«Der Buchhändler, Sir -- wollen wir nicht zu ihm?» fiel Oliver ein.
«Ich weiß, wo er wohnt. O bitte, lassen Sie uns zu ihm fahren.»

«Mein liebes Kind, wir haben für einen Tag der Täuschung genug gehabt»,
erwiderte Losberne. «Führen wir zum Buchhändler, so würden wir sicher
hören, daß er sein Haus angezündet hat, oder davongegangen oder
tot wäre. Nein, wir wollen für heute sogleich wieder nach Chertsey
zurückkehren.»

Er wiederholte, gemäß dem Eindruck des Augenblicks, seinen Befehl, und
sie kehrten nach Chertsey zurück.

Die erfahrene bittere Täuschung verursachte Oliver mitten in seinem
Glücke viel Kummer; denn wie oft hatte er sich während seiner Krankheit
an der Vorstellung gelabt, was Mr. Brownlow und Mrs. Bedwin zu ihm
sagen würden, und welche Wonne es sein müßte, ihnen zu erzählen, wie
viele lange Tage und Abende er zugebracht in der Erinnerung an das, was
sie für ihn getan, und in Tränen über seine schreckliche Entführung
aus ihrem Hause. Die Hoffnung, sich von Verdacht bei ihnen reinigen zu
können, hatte ihn in mancher bösen Stunde aufrecht erhalten; und nun
war der Gedanke, daß sie außer Landes gegangen in dem Glauben, daß er
ein Dieb und Betrüger sei -- einem Glauben, in welchem sie vielleicht
bis zu ihrer Sterbestunde verharrten -- fast mehr, als er zu ertragen
vermochte.

Das Benehmen seiner Wohltäter und Gönner gegen ihn blieb jedoch
unverändert. Als nach vierzehn Tagen schönes Frühlingswetter war, die
Bäume im jungen, frischen Grün zu prangen und die Blumen zu blühen
anfingen, trafen sie die erforderlichen Vorbereitungen, ihre Wohnung
in Chertsey auf einige Monate zu verlassen. Das Silbergerät, das die
Begierde des Juden erregt hatte, wurde in sicheren Gewahrsam gebracht,
Giles mit einem zweiten Diener zur Bewachung des Hauses zurückgelassen,
und sie reisten ab auf das Land und nahmen Oliver mit.

Wer vermöchte das selige Entzücken, den Seelenfrieden und die süße,
trauliche Ruhe zu schildern, die der noch immer schwache Knabe in der
balsamischen Luft, auf den grünen Hügeln und in den schönen Waldungen
empfand, die das kleine Dorf, seinen neuen Wohnsitz, umgaben! Wer
könnte es mit Worten beschreiben, welche Stille, welche Frische, welche
Lust ein Frühling auf dem Lande in die Herzen geplagter Stadtbewohner
senkt! Selbst von Leuten, die in engen, menschengefüllten Straßen
ihr Leben unter stetem Geräusch und in fortwährender Plackerei
zugebracht haben, und in denen nie ein Wunsch nach Veränderung ihrer
Lage aufgestiegen ist, und die das Mauerwerk und die Steine, die
engen Grenzmarken ihrer kleinen, täglichen Ausflüge, fast zu lieben
angefangen -- selbst von ihnen, wenn die Todesstunde sich ihnen nahte,
weiß man es, daß sie sich endlich nach einem flüchtigen Blicke des
Antlitzes der Natur sehnten, daß sie, hinweggeführt von dem Schauplatze
ihrer Mühen, Schmerzen und Freuden, gleichsam verjüngt zu werden
schienen, Tag für Tag ein grünes, sonniges Plätzchen aufsuchten und
in dem bloßen Schauen des blauen Himmels, der blumenübersäten Wiesen
und des glitzernden Stromes einen Vorgeschmack des Himmels selbst
empfanden, der ihre letzten Leiden versüßte, so daß sie friedlich wie
die untergehende Sonne in ihre Gräber sanken, gleich der Sonne, die sie
mit Entzücken am Fenster ihres einsamen, stillen Kämmerchens sinken
sahen. Die Erinnerungen, welche durch friedliche ländliche Szenen
hervorgerufen werden, sind nicht von dieser Welt und ihren Gedanken
oder Hoffnungen. Ihr süßes, lindes Einwirken kann uns lehren, frische
Kränze für die Gräber unserer Lieben zu winden, unsere Herzen zu
läutern und unseren alten Haß, unsere Feindschaften zu verscheuchen und
auszutilgen; und durch das alles zieht sich auch bei minder sinnigen
Gemütern ein halbes, unbestimmtes Bewußtsein, Gefühle solcher Art einst
in einer fernen, längstentflohenen Zeit empfunden zu haben -- ein
Bewußtsein, das feierlich-ernste Ahnungen einer entfernten kommenden
Zeit erweckt, und Stolz und Weltsinn dämpft und unterdrückt.

Das Dörfchen, wohin sie sich begaben, lag äußerst angenehm, und Oliver
war es, als wenn ein neues Leben für ihn begonnen hätte, denn er
hatte seine Tage von frühester Kindheit an in engen, oft schmutzigen
Räumen und unter Geräusch und Lärm zugebracht. Rosen und Geißblatt
bedeckten die Wände des Häuschens seiner Gönnerin, die Stämme der Bäume
waren mit Efeu bewachsen, und die Gartenblumen erfüllten die Luft mit
köstlichen Düften. Dicht neben dem Häuschen lag ein kleiner Friedhof,
nicht angefüllt mit hohen, widerwärtigen Grabsteinen, sondern voll von
bescheidenen Gras- und Mooshügelchen, unter welchen die alten Leute des
Dorfes von ihren Mühen ausruhten. Oliver besuchte ihn oft und setzte
sich, des elenden Grabes seiner Mutter gedenkend, bisweilen nieder und
weinte ungesehen; doch wenn er dann die Augen emporhob zu dem klaren
blauen Himmel über ihm, so dachte er sie sich nicht mehr ruhend im
Schoße der Erde, sondern droben in den Wohnungen der Seligen und weinte
wohl fort um sie, doch ohne Schmerz.

Es war eine schöne, glückliche Zeit. Die Tage vergingen friedlich
und heiter, und die Abende brachten weder Furcht noch Sorge, kein
Schmachten in einem düsteren Kerker, nicht den Anblick heimkehrender,
verworfener Menschen mit sich, sondern nur süße, traute Gedanken. Jeden
Morgen ging Oliver zu einem silberhaarigen, alten Manne, der dicht
neben der kleinen Kirche wohnte und ihn lesen und schreiben lehrte,
und so freundlich mit ihm redete und sich so sehr um ihn bemühte, daß
Oliver sich selbst nie genug tun konnte, ihm Freude zu machen. Zu
anderen Tagesstunden lustwandelte er mit Mrs. Maylie und Miß Rose und
hörte ihrer Unterhaltung zu oder saß bei ihnen an einem schattigen
Plätzchen und horchte dem Vorlesen der jüngeren Dame, ohne sich jemals
satthören zu können. Zu anderen Zeiten war er eifrig mit seiner
Lektion auf den folgenden Tag in einem kleinen Zimmer beschäftigt,
dessen Fenster in den Garten ging; und wenn der Abend herankam, ging
er wieder mit den Damen aus und war überglücklich, wenn er ihnen eine
Blume pflücken konnte, nach welcher sie etwa Begehrung trugen, oder
wenn sie etwas vergessen hatten und ihm auftrugen, es zu holen. War es
dämmerig geworden, so pflegte sich Rose an das Fortepiano zu setzen
und zu spielen oder ein altes Lied zu singen, das ihre Tante zu hören
wünschte, und Oliver saß dann am Fenster und horchte den lieblichen
Tönen, und Zähren wehmütiger Lust rannen über seine Wangen hinab.

Wie ganz anders wurde der Sonntag hingebracht, als ihn Oliver je
verlebt hatte, und welch ein schöner Tag war er gleich den anderen
Tagen in dieser glücklichen Zeit! Morgens wurde die kleine Kirche
besucht, vor deren Fenstern sich grüne Blätter im Winde bewegten,
und draußen zwitscherten die Vögel, und durch die niedrige Tür drang
die reine, erquickende Luft herein. Die armen Landleute erschienen
so sauber und reinlich und knieten bei den Gebeten so ehrfurchtsvoll
nieder, daß ihr Gottesdienst wie eine Freude und nicht wie eine
beschwerliche Pflichtübung erschien; und wenn der Gesang auch weniger
als kunstlos war, so kam er doch vom Herzen und klang zum wenigsten
Olivers Ohre wohltönender als alle Kirchenmusik, die er in seinem
ganzen Leben gehört hatte. Und dann wurden die Spaziergänge wie
gewöhnlich gemacht, und manche reinliche Hütte im Dorfe ward besucht;
abends las dann Oliver einige Kapitel aus der Bibel vor, die ihm in der
Woche vorher erklärt waren, und er empfand dabei eine so stolze Freude,
als wenn er der Geistliche selbst gewesen wäre.

Morgens früh um sechs Uhr war er auf und draußen und streifte in den
Feldern umher, Sträuße von wilden Blumen pflückend, mit denen er den
Frühstückstisch schmückte. Auch brachte er frisches Kreuzkraut für
Roses Vögel mit nach Hause, und waren dieselben besorgt, so hatte er
fast täglich einen kleinen Mildtätigkeitsauftrag im Dorfe auszurichten,
oder es war etwas im Garten zu tun, wobei er unter der Anleitung des
Gärtners den lebhaftesten Eifer bewies, bis Miß Rose erschien und ihn
durch manches Lächeln, manchen freundlichen Lobspruch belohnte.

So vergingen drei Monate -- drei Monate, die im Leben der Glücklichsten
schön zu nennen gewesen sein würden, für Oliver aber, nach seinen
unruhigen, trüben Tagen, die ungemischteste Seligkeit waren. Bei
reinster und edelster Liebe und Großmut auf der einen und bei der
wahrhaft innigsten und wärmsten Dankbarkeit auf der anderen Seite
war es in der Tat kein Wunder, daß Oliver am Schlusse dieses kurzen
Zeitabschnittes bei der alten Dame und ihrer Nichte vollkommen heimisch
geworden war, und daß beide durch ihren Stolz auf ihn und ihre Freude
an ihm die heiße Zuneigung seines jungen und lebhaft empfänglichen
Herzens vergalten.




33. Kapitel.

    In dem Olivers und seiner Gönnerinnen Glück eine plötzliche Störung
    erleidet.


Der Frühling schwand rasch dahin, und der Sommer kam, und war alles
umher schön gewesen im Lenz, so blühte und glänzte es nun in vollster,
üppigster Pracht. Die Bäume streckten ihre Arme über den durstigen
Boden aus, verwandelten offene und nackte Stellen in dunkle, heimliche
Plätzchen, und wie köstlich ließen sich aus ihrem stillen, hehren
Schatten die sonnigen Felder beschauen! Die Erde hatte sich mit ihrem
glanzvoll grünsten Mantel geschmückt, und Millionen Blüten durchdufteten
die Luft. Alles grünte, blühte, strahlte von Lust und verkündete Freude.

Das ruhige Leben in Mrs. Maylies Landhäuschen nahm seinen Fortgang,
und heiter und froh genossen die Bewohner die schöne Zeit. Oliver
war gesund und kräftig geworden, ohne daß -- wie es sonst wohl der
Fall ist -- eine Änderung in seinen Gefühlen oder seinem Benehmen
eingetreten wäre. Er war fortwährend derselbe sanfte, zärtliche,
liebevolle Knabe, der er gewesen, als unter Krankheit und Schmerz seine
Kräfte geschwunden waren und seine Schwäche ihn auch bei den kleinsten
Wünschen und Bedürfnissen von seinen Pflegerinnen abhängig gemacht
hatte.

Einst an einem schönen Abende machte er mit Mrs. Maylie und Rose
einen ungewöhnlich langen Spaziergang; es war sehr heiß gewesen, doch
kühlte jetzt ein linder Wind die Luft, und am Himmel glänzte schon
der Vollmond. Rose war sehr munter und wohlgemut, sie gingen unter
fröhlichem Gespräche weiter, als sie gewöhnlich zu tun pflegten, Mrs.
Maylie empfand endlich Ermüdung, und sie kehrten langsamer nach Hause
zurück. Rose legte nur ihren Hut ab, setzte sich wie gewöhnlich an das
Piano, schlug einige Akkorde an, ging zu einer langsam-feierlichen
Weise über und fing, während sie dieselbe spielte, zu schluchzen an.

«Was weinst du, liebes Kind?» fragte Mrs. Maylie; allein Rose
antwortete nicht und spielte nur ein wenig rascher, als wenn sie aus
einem schmerzlichen Sinnen aufgeweckt worden wäre.

«Liebes Kind, was ist dir?» fragte Mrs. Maylie, hastig aufstehend und
sich über sie beugend. «Dein Gesicht ist in Tränen gebadet. Was betrübt
dich denn, bestes Kind?»

«Nichts, Tante, nichts», erwiderte Rose. «Ich weiß selbst nicht, wie
mir ist -- ich kann es nicht beschreiben -- ich fühle mich so matt, so
--»

«Du bist doch nicht krank, Rose?» fiel Mrs. Maylie ein.

«O nein, nein», sagte die junge Dame schaudernd, als wenn sie plötzlich
von einem Fieberfroste geschüttelt würde; «mir wird wenigstens sogleich
wieder besser sein. Schließe das Fenster, Oliver.»

Oliver eilte, ihr Geheiß zu erfüllen, sie zwang sich, heiter zu
scheinen und spielte eine muntere Weise; allein die Hände fielen
ihr kraftlos in den Schoß, sie stand auf, sank auf das Sofa nieder,
bedeckte ihr Antlitz und ließ den Tränen freien Lauf, die sie nicht
mehr zu unterdrücken vermochte.

«Mein liebes Kind!» rief Mrs. Maylie, sie an die Brust drückend, aus;
«ich habe dich ja noch nie so gesehen!»

«Ich beunruhige Sie nur sehr ungern,» erwiderte Rose, «kann aber trotz
aller Mühe dies Weinen nicht unterdrücken. Ich fürchte, daß ich doch
krank bin, Tante.»

Sie war es in der Tat; denn als Licht gebracht wurde, gewahrten alle,
daß sich ihre Farbe in der kurzen Zeit seit der Rückkehr von dem
Spaziergange in Marmorblässe verwandelt hatte. Ihr Antlitz hatte nichts
von seiner Schöne verloren, und doch war mit ihren Zügen eine Wandlung
vorgegangen, und es lag ein Ausdruck der Unruhe und Abspannung darin,
den sie noch niemals gezeigt hatten. Nach Verlauf einer Minute waren
ihre Wangen wieder von Purpurröte übergossen, ihre sanften, blauen
Augen bekamen einen stechenden, unheimlichen Blick, und auch dieser
verschwand bald wieder, gleich einem vorüberziehenden Wölkchen, und die
Leichenblässe kehrte zurück.

Oliver, der die alte Dame genau beobachtet hatte, bemerkte, daß sie
große Unruhe empfand, wie es in Wahrheit bei ihm selber der Fall war;
da sie sich indes offenbar den Anschein zu geben suchte, als wenn sie
die Sache leicht nähme, so tat er dasselbe, was bei Rose eine günstige
Wirkung hervorzubringen schien. Denn als sie auf Zureden ihrer Tante zu
Bett ging, sah sie wieder wohler aus, versicherte, es auch zu sein und
fügte hinzu, sie wäre überzeugt, daß sie am anderen Morgen gesund und
munter wie sonst erwachen würde.

«Ich hoffe, Ma'am,» sagte Oliver, als Mrs. Maylie zurückkehrte, «daß
Miß Rose nicht ernstlich krank werden wird. Sie sah heute abend unwohl
genug aus; doch --»

Die alte Dame winkte ihm, nicht fortzufahren, setzte sich, stützte
schweigend den Kopf auf die Hand und sagte endlich mit bebender Stimme:
«Ich will es auch hoffen, Oliver. Ich habe einige Jahre sehr glücklich
-- vielleicht zu glücklich mit ihr verlebt, und es könnte Zeit sein,
daß mir wieder ein Unglück begegnet -- ich hoffe indes, nicht dieses.»

«Was für ein Unglück, Ma'am?» fragte Oliver.

«Ich meine den schweren Schlag,» antwortete die alte Dame fast tonlos,
«das liebe Mädchen zu verlieren, das so lange schon meine Freude und
mein Trost gewesen ist.»

«Das verhüte Gott!» rief Oliver hastig aus.

«Ich sage Ja und Amen dazu, mein Kind!» fiel die alte Dame, die Hände
ringend, ein.

«Sie brauchen sicher so etwas Schreckliches nicht zu fürchten», fuhr
Oliver fort. «Miß Rose war ja vor zwei Stunden vollkommen wohl.»

«Und jetzt ist sie sehr unwohl», versetzte Mrs. Maylie, «und wird ohne
Zweifel noch kränker werden. O meine liebe, liebe Rose! Was sollte ich
anfangen ohne sie!» Sie wurde so sehr und so schmerzlich bewegt, daß
Oliver, seine eigene Herzensangst unterdrückend, sich bemühte, sie
zu beruhigen, und sie dringend bat, um der lieben jungen Dame selbst
willen gefaßter zu sein.

«Bedenken Sie doch nur, Ma'am,» sagte er, gewaltsam die Tränen
zurückdrängend, die ihm in die Augen schossen, «wie jung und wie gut
sie ist und wie sie alles um sich her erfreut. Ich weiß es -- weiß
es ganz gewiß, daß sie um ihrer selbst und um Ihret- und unser aller
willen, die sie so froh und glücklich macht, nicht sterben wird; nein,
nein, Gott läßt sie nimmermehr schon jetzt sterben!»

«Ach! du sprichst und denkst wie ein Kind, mein guter Oliver,» sagte
Mrs. Maylie, ihm die Hand auf den Kopf legend, «und irrst, so natürlich
es sein mag, was du sagst. Indes hast du mich an meine Pflicht
erinnert. Ich hatte sie auf einen Augenblick ganz vergessen, Oliver,
und hoffe Verzeihung zu finden, denn ich bin alt und habe genug gesehen
von Krankheiten und vom Tode, um den Schmerz zu kennen, den sie den
Hinterbleibenden zufügen. Auch besitze ich genug Erfahrung, um zu
wissen, daß es nicht immer die Jüngsten und Besten sind, die den sie
Liebenden erhalten werden -- was uns jedoch eher trösten als bekümmern
sollte, denn der Himmel ist weise und gütig, und Erfahrungen solcher
Art lehren uns eindringlich, daß es eine noch schönere Welt gibt als
diese und daß wir bald hinübergehen zu ihr. Gottes Wille geschehe! Doch
liebe ich sie, und er allein weiß es, wie sehr, wie sehr!»

Oliver war verwundert, daß Mrs. Maylie, sobald sie diese Worte
gesprochen, ihren Klagen plötzlich Einhalt tat, sich hoch emporrichtete
und vollkommen ruhig und gefaßt erschien. Er war noch mehr erstaunt,
als er bemerkte, daß sie sich in ihrer Festigkeit gleich, bei allem
Sorgen und Wachen besonnen und gesammelt blieb und jede ihrer Pflichten
dem Anscheine nach sogar mit Heiterkeit erfüllte. Doch er war jung und
wußte noch nicht, welch großen Tuns und Duldens starke Seelen unter
schwierigen und entmutigenden Umständen fähig sind; und wie hätte er es
wissen sollen, da sich die Starken selbst ihrer Kraft nur selten bewußt
sind?

Es folgte eine angstvolle Nacht, und als der Morgen kam, waren Mrs.
Maylies Vorhersagungen nur zu wahr geworden. Rose lag im ersten Stadium
eines heftigen und gefahrdrohenden Fiebers.

«Wir müssen tätig sein, Oliver, und dürfen uns nicht einem nutzlosen
Schmerze überlassen», sagte Mrs. Maylie, den Finger auf den Mund legend
und ihm fest in das Gesicht blickend. «Dieses Schreiben muß so eilig
wie irgend möglich Mr. Losberne zugeschickt werden. Du sollst es nach
dem Flecken tragen, der auf dem Fußwege nur vier Meilen entfernt ist;
von dort soll ein reitender, expresser Bote nach Chertsey abgehen.
Der Gastwirt besorgt ihn, und ich weiß, daß du den Auftrag pünktlich
ausrichten wirst.»

Oliver konnte nicht antworten, allein seine Mienen verkündigten, daß er
vor Begierde brannte, sich sogleich auf den Weg zu begeben.

«Hier ist noch ein Schreiben,» fuhr Mrs. Maylie nachsinnend
fort, «allein ich weiß kaum, ob ich es sogleich abschicken oder
abwarten soll, wie es mit Roses Befinden wird. Ich möchte es lieber
zurückhalten, bis ich das Schlimmste fürchten müßte.»

«Soll er auch nach Chertsey, Ma'am?» fragte Oliver ungeduldig,
seinen Auftrag auszurichten, und die zitternde Hand nach dem Briefe
ausstreckend.

«Nein!» erwiderte die alte Dame.

Sie gab ihn jedoch dem Knaben, da sie in Gedanken verloren war, und
Oliver sah, daß er an Harry Maylie Esq. und nach dem Landsitze eines
Lords, dessen Namen er noch nie gehört hatte, adressiert war.

«Soll er fort, Ma'am?» fragte Oliver ungeduldig.

«Nein; ich will bis morgen warten», sagte die alte Dame, ließ sich das
Schreiben zurückgeben, reichte Oliver ihre Börse, und er eilte hinaus,
um in kürzester Frist nach dem Marktflecken zu gelangen, in welchem
er staubbedeckt ankam. Er hatte bald das Gasthaus zum Georg gefunden
und wandte sich an einen Postillon, der ihn an den Hausknecht verwies,
von welchem er wiederum an den Wirt verwiesen wurde, der bedächtig zu
lesen und dann zu schreiben und Befehle zu erteilen anfing, worüber
manche Minute verging. Oliver hätte selbst auf das Pferd springen und
davongaloppieren mögen; doch endlich sprengte ein Berittener des Wirts
die Straße hinunter und war nach wenigen Augenblicken verschwunden.
Oliver, der vor der Tür gestanden hatte, ging mit leichterem Herzen
über den Hof des Gasthauses, um eiligst heimzukehren. Als er um die
Ecke eines Stallgebäudes bog, rannte er gegen einen großen, in einen
Mantel eingehüllten Mann an, der eben aus der Tür des Gasthauses
getreten sein mußte.

«Ha! zum Teufel, was ist das?» rief der Mann zurückprallend und die
Blicke auf Oliver heftend.

«Ich bitte um Vergebung, Sir», sagte Oliver; «ich hatte große Eile und
sah Sie nicht kommen.»

«Alle Teufel!» murmelte der Mann vor sich hin, den Knaben mit seinen
großen, schwarzen Augen anstarrend. «Wer hätte das denken können? Und
wenn man ihn zu Staub zerriebe, er würde aus 'nem marmornen Sarge
wieder aufstehen und mir in den Weg treten.»

«Es tut mir leid, Sir», stotterte Oliver verwirrt; «ich hoffe, daß ich
Ihnen keinen Schaden getan habe.»

«Daß seine Knochen verfaulen!» murmelte der finstere Mann durch
die verbissenen Zähne; «hätte ich nur den Mut gehabt, das Wort
auszusprechen, so hätte mich eine einzige Nacht von ihm befreien
können. Fluch über dein Haupt und die Pest in deinen Leib, du
Höllenbrand! Was hast du hier zu schaffen?»

Er hob drohend die Faust empor, knirschte mit den Zähnen und trat einen
Schritt vor, als wenn er Oliver einen Schlag versetzen wollte, stürzte
aber plötzlich zu Boden und wand und krümmte sich, während ihm dicker
Schaum vor dem Munde stand. Oliver schaute dem Wahnwitzigen (denn ein
solcher schien ihm der schreckliche Mann zu sein) ein paar Augenblicke
zu, lief darauf in das Haus, um Beistand zu holen, verlor sodann
keine Zeit mehr und eilte nach Hause zurück, mit großer Verwunderung
und nicht ohne Bangigkeit an das seltsame Benehmen des Unbekannten
zurückdenkend. Er verlor den ganzen Vorfall jedoch bald aus dem
Gedächtnis, denn als er in Mrs. Maylies Wohnung wieder angelangt war,
hörte und sah er genug, was seinen Gedanken eine ganz andere Richtung
gab.

Roses Zustand hatte sich sehr verschlimmert, und noch vor Mitternacht
lag sie in Fieberphantasien. Der Wundarzt aus dem Dorfe hatte
Mrs. Maylie erklärt, daß die Krankheit ihrer Nichte eine sehr
beunruhigende Wendung genommen hätte, und zwar in dem Maße, daß ihre
Wiederherstellung einem Wunder gleichkommen würde.

Wie oft sprang Oliver aus seinem Bett in der Schreckensnacht, um an
die Treppe zu schleichen und zu horchen, was in dem Krankenzimmer
vorgehen möchte! Er bebte fast fortwährend an allen Gliedern, und kalte
Schweißtropfen traten ihm auf die Stirn, wenn ihm irgendein Geräusch zu
verkünden schien, daß das Schlimmste eingetreten sei. Er hatte nie so
inbrünstig zum Himmel gefleht, wie er in dieser Nacht um die Erhaltung
des teuren Lebens seiner holden, am Rande des Grabes stehenden Freundin
betete.

Die Ungewißheit, die schreckliche, ängstigende Ungewißheit, wenn wir
untätig daneben stehen, während die Wagschale eines Heißgeliebten
zwischen Tod und Leben schwankt -- die folternden Gedanken, welche dann
auf das Gemüt einstürmen, das Herz zu rascheren, heftigen Schlägen
treiben, den Atem stocken machen -- die düsteren Bilder, welche sie
heraufbeschwören --, der verzweifelte Herzensdrang, etwas zu tun zur
Linderung von Schmerzen, die wir nicht lindern können, zur Entfernung
einer Gefahr, die wir nicht zu entfernen vermögen, und die tiefe,
traurige Niedergeschlagenheit, welche uns dann bei dem Bewußtsein
unserer Ohnmacht ergreift: -- welche Qualen lassen sich diesen
vergleichen, durch welche Erwägungen oder Anstrengungen könnten wir sie
uns in der Fieberhitze der Aufregung, in unserer tiefen Not erleichtern?

Der Morgen kam, und das Häuschen war stumm und still. Man flüsterte
nur; von Zeit zu Zeit ließen sich angstvolle Gesichter an der Tür
blicken, und Frauen und Kinder gingen weinend wieder fort. Den ganzen
langen Tag und noch stundenlang, nachdem es dunkel geworden war,
ging Oliver leise im Garten auf und ab, die Augen fortwährend hinauf
nach dem Zimmer der Kranken gewandt und schaudernd beim Anblick des
verdunkelten Fensters, das ihm aussah, als wenn drinnen der Tod lauernd
ausgestreckt läge. Zu einer späten Abendstunde traf Mr. Losberne ein.
«'s ist hart», sagte der weichherzige Doktor, sich abwendend; «'s ist
hart -- so jung -- so heiß geliebt von so vielen --, doch aber ist nur
wenig Hoffnung!»

An einem abermaligen Morgen strahlte die Sonne hell -- so hell und
heiter, als wenn sie auf kein Leiden, keine Sorge herabblickte; und
indem die Blumen sie umblühten und Leben, Gesundheit und Töne der
Freude und lachende Gegenstände sie rings umgaben, siechte die junge,
schöne Dulderin dem Grabe entgegen. Oliver schlich hinaus auf den
stillen Friedhof, setzte sich auf einen der kleinen, grünen Hügel und
weinte um sie in der Stille und Einsamkeit.

Der Tag war ein so köstlicher Sommertag, die sonnige Landschaft so
heiter und glänzend, die Vögel sangen und hüpften so munter in den
Zweigen oder schwangen sich so lebensfroh in die Lüfte empor, alles,
alles schien so laut aufzufordern zur Freude und Lust, daß sich dem
Knaben, als er die schmerzenden Augen aufschlug, unbewußt der Gedanke
aufdrängte, dies sei keine Zeit für den Tod, und Rose könne nimmermehr
sterben, während so viele weit geringere Wesen so froh und munter
wären; die Gräber wären nur für den kalten, freudlosen Winter, nicht
für die sonnige, duftige, Lust weckende und gebende Sommerzeit. Fast
hätte er geglaubt, die Leichentücher wären für die Alten und Abgelebten
und nicht dazu bestimmt, die jungen und schönen Gestalten mit ihrer
grausigen Nacht zu bedecken.

Ein Geläute der Kirchglocke unterbrach plötzlich seine kindlichen
Gedanken. Es wurde zu den Begräbnisgebeten geläutet. Ein ländliches
Leichengefolge schritt durch das Tor herein; die Leidtragenden hatten
sich mit weißen Schleifen geschmückt; sie begruben einen Jüngling.
Sie standen mit entblößten Häuptern am Grabe, und in ihrer Mitte
kniete eine weinende Mutter. Aber die Sonne schien hell, und die Vögel
zwitscherten und hüpften in den Zweigen fort und fort.

Oliver kehrte nach Hause zurück, gedenkend der vielfachen Beweise von
Güte, die er von der jungen Dame empfangen und mit dem Wunsche, daß
die Zeit noch einmal kommen möchte, wo er imstande wäre, ihr ohne
Aufhören zu zeigen, wie dankbar und liebevoll gesinnt er gegen sie war.
Er hatte sich nichts vorzuwerfen, denn er war eifrig in ihrem Dienste
gewesen, und doch mußte er an zehn und wieder zehn Fälle denken, in
welchen er meinte, nicht eifrig genug gewesen zu sein. Wohl sollten wir
sorgfältig über unser Benehmen gegen die, mit denen unsere Lebensbahn
uns zusammenführt, wachen, und so viel Liebe als möglich hineinlegen;
denn jeglichen Todesfall begleitet eine Schar von Gedanken an so viel
Versäumtes, so wenig Getanes -- an so viel Vergessenes und an noch viel
mehr, was hätte besser getan, oder wieder gut gemacht werden können,
daß die Erinnerungen dieser Art zu den allerbittersten gehören, die uns
quälen können. Keine Reue ist so schmerzlich, als die vergebliche, und
wollen wir uns ihre Peinigungen ersparen, so laßt uns beizeiten allen
dessen gedenken.

Als Oliver zu Hause angelangt war, fand er Mrs. Maylie im kleinen
Wohnzimmer. Sein Herz zagte in ihm bei ihrem Anblick, denn sie hatte
das Bett ihrer Nichte noch keine Minute verlassen, und er zitterte, zu
denken, welche Veranlassung sie von demselben verscheucht haben könnte.
Er vernahm, daß die Patientin in einen festen Schlummer verfallen
sei, aus welchem sie erwachen würde zur Genesung und zum Leben, oder
um ihren Lieben das letzte Lebewohl zu sagen und von dieser Welt zu
scheiden.

Sie saßen stundenlang horchend und zu sprechen sich scheuend,
beieinander. Das Mahl wurde unangerührt hinausgetragen, ihre Blicke
hingen an der Pracht der untergehenden Sonne, doch waren ihre Gedanken
bei einem anderen Gegenstande. Ihr gespanntes Ohr vernahm den Schall
herannahender Fußtritte, und sie eilten zugleich nach der Tür, als
Losberne eintrat.

«Was haben Sie von Rose zu melden?» rief ihm die alte Dame entgegen.
«Sagen Sie es sogleich. Ich kann alles, nur keine Ungewißheit ertragen.
In des Himmels Namen, reden Sie! Ist sie tot, ist sie tot?»

«Nein», entgegnete der Doktor äußerst bewegt. «So wahr er gütig und
barmherzig ist, wird sie leben, um uns alle noch viele Jahre zu
beglücken!»

Die alte Dame fiel auf die Knie nieder und mühte sich, die Hände zu
falten; allein ihre Kraft, die sie so lange aufrecht erhalten hatte,
floh mit dem ersten Dankesseufzen, das sie zum Himmel emporsandte, und
sie sank zurück in die Arme des herbeigeeilten Doktors.




34. Kapitel.

    In welchem ein junger Herr auftritt, und Oliver ein neues Abenteuer
    erlebt.


Es war fast zu viel Glück, um es ertragen zu können. Oliver war durch
die unverhoffte Kunde ganz betäubt; er konnte nicht weinen, nicht
sprechen, nicht bleiben, wo er war. Er mußte sich erst wieder zu fassen
suchen, um was er gehört, zum klaren Bewußtsein zu bringen, als er
sich nach einem langen Umherschweifen in der stillen Abendlandschaft
durch einen Tränenstrom erleichtert, und von der fast nicht mehr zu
ertragenden Last befreit fühlte, die ihm gleich einem Alp auf dem
Herzen gelegen hatte.

Es dunkelte, als er nach Hause zurückkehrte, beladen mit Blumen, die
er mit ungewöhnlicher Sorgfalt zur Ausschmückung des Krankenzimmers
gepflückt hatte. Als er der Wohnung Mrs. Maylies rasch zuschritt, hörte
er hinter sich auf der Straße das donnernde Geräusch eines Wagens.
Er sah sich um: es war eine Postchaise, und da die Straße ziemlich
schmal war und der Postillon im Galopp fuhr, so trat er dicht an ein
Gartentor, um nicht in Gefahr zu geraten. Die Chaise näherte sich, und
nun erblickte er ein unter einer Nachtmütze fast verstecktes Gesicht,
das ihm bekannt schien; er begann nachzusinnen, wem es angehören
möchte, als er angerufen wurde, und der Postillon den Befehl zum Halten
erhielt.

«Oliver, wie steht es -- wie steht es mit Miß Rose, Oliver?» rief ihm
Mr. Giles zu.

«Ohne Umschweife -- besser oder schlimmer?» rief ein junger Herr, der
Giles zurückzog und sich selbst aus dem Schlage herausbeugte.

«Besser -- viel besser!» erwiderte Oliver mit freudiger Hast.

«Gott sei Dank!» rief der junge Herr aus. «Ist's auch gewiß?»

«Sie können sich fest darauf verlassen, Sir», sagte Oliver; «die
Besserung trat vor ein paar Stunden ein, und Mr. Losberne hat gesagt,
daß alle Gefahr vorüber sei.»

Der junge Mann sagte kein Wort mehr, sondern sprang aus dem Wagen, zog
Oliver zur Seite und fragte ihn mit bebender Stimme: «Ist es auch ganz
gewiß? -- irrst du auch nicht, Kleiner? Täusche mich nicht, indem du
Hoffnungen in mir erweckst, die am Ende nicht in Erfüllung gehen.»

«Das möcht' ich um keinen Preis, Sir», erwiderte Oliver. «Sie können
mir in der Tat glauben. Mr. Losbernes Worte waren, sie würde leben
und uns alle noch viele Jahre beglücken. Ich hab' es ihn selbst sagen
hören.»

In seinen Augen standen Tränen, während er sich an die Worte erinnerte,
die ihn so unaussprechlich glücklich gemacht hatten, und der junge Herr
wandte das Gesicht ab und war einige Minuten stumm. Oliver glaubte ihn
schluchzen zu hören und wagte es nicht, seinen Bericht fortzusetzen; er
stand da und tat, als wenn er mit seinem Blumenstrauß beschäftigt wäre.

Mr. Giles hatte unterdes auf dem Kutschtritte, die Ellenbogen auf die
Knie gestützt und die Augen trocknend, gesessen, und die Röte der
letzteren, als der junge Herr ihn anredete, und als er aufblickte,
bewies, daß seine Bewegung keine erkünstelte war.

«Fahren Sie nach meiner Mutter Hause, Giles», sagte der junge Herr.
«Ich will langsam nachkommen, um mich erst ein wenig zu sammeln, bevor
ich ihr unter die Augen trete. Sie können ihr sagen, daß ich käme.»

«Bitt' um Vergebung, Mr. Harry,» erwiderte Giles, «aber Sie würden mir
einen großen Gefallen erzeigen, wenn Sie sich durch den Postillon
anmelden lassen wollten. Die Damen dürfen mich wirklich so nicht sehen,
Sir; ich würde alles Ansehen bei ihnen verlieren.»

«Nach Ihrem Belieben, Giles», entgegnete der junge Herr lächelnd.
«Lassen Sie ihn mit dem Gepäck vorausfahren, und Sie können mit uns
nachfolgen, nur vertauschen Sie jetzt sogleich Ihre Nachtmütze mit
einer angemessenen Kopfbedeckung, damit wir nicht für Wahnwitzige
gehalten werden.»

Giles erinnerte sich mit Schrecken seines unziemlichen Aufzugs, steckte
seine Nachtmütze in die Tasche, setzte statt derselben einen Hut auf,
der Postillon fuhr weiter, und Giles, Mr. Maylie und Oliver folgten zu
Fuß nach.

Oliver blickte den jungen Herrn von Zeit zu Zeit mit ebensoviel Neugier
wie Interesse von der Seite an. Mr. Maylie schien etwa fünfundzwanzig
Jahre alt zu sein, und war von Mittelgröße; in seinem wohlgeformten
Gesicht drückte sich Offenheit aus, und sein Benehmen war äußerst
gewandt und gewinnend. Trotz der Altersverschiedenheit sah er der alten
Dame so sprechend ähnlich, daß ihn Oliver sogleich als einen nahen
Anverwandten derselben erkannt haben würde, wenn er sie auch nicht
seine Mutter genannt hätte.

Mrs. Maylie erwartete ihn mit großer Sehnsucht und Ungeduld, und das
Wiedersehen der Mutter und des Sohnes fand nicht ohne Bewegung statt.

«O Mutter, warum schrieben Sie mir nicht früher?» flüsterte er.

«Ich schrieb allerdings,» erwiderte sie, «beschloß aber nach reiflicher
Überlegung, den Brief zurückzuhalten, bis ich Mr. Losbernes Ausspruch
gehört haben würde.»

«Aber warum setzten Sie sich einer Gefahr aus, deren Eintreten so sehr
möglich war? Wenn Rose -- ich kann das Wort jetzt nicht aussprechen --
wenn Roses Krankheit eine andere Wendung genommen, wie hätten Sie sich
jemals selbst verzeihen können -- wie hätte ich je wieder ruhig werden
sollen?»

«Wenn das Schlimmste eingetreten wäre, Harry, so fürchte ich, daß
deine Ruhe sehr wesentlich gestört worden und daß es von nur sehr
geringer Bedeutung gewesen sein würde, ob du hier einen Tag früher oder
später eingetroffen wärest.»

«Sie müssen es am besten wissen, und jedenfalls leidet das keinen
Zweifel, daß meine Ruhe, wenn das Schlimmste eingetreten wäre --»

«Rose verdient die echteste, reinste Neigung, die das Herz eines
Mannes nur bieten kann. Ihr Seelenadel und ihr liebendes, hingebendes
Gemüt rechtfertigen den Anspruch auf eine nicht gewöhnliche, sondern
tiefe und dauernde Gegenliebe. Wenn ich davon nicht überzeugt wäre und
nicht außerdem wüßte, daß ein verändertes Benehmen von seiten eines
Anverwandten, den sie liebt, sie bis zum Tode betrüben würde, so würde
mir meine Aufgabe nicht so schwierig erscheinen, oder ich hätte nicht
so viele Kämpfe mit mir selbst zu bestehen, indem ich tue, was mir die
Pflicht schlechterdings zu gebieten scheint.»

«Ist das nicht unrecht, Mutter? Halten Sie mich noch für so jung,
daß ich mein Herz nicht kennte, imstande wäre, meine innersten,
lebhaftesten, besten Gefühle zu mißdeuten?»

«Mein lieber Harry, die Jugend hegt viele edle Gefühle, welche nicht
von Dauer und bisweilen, wenn befriedigt, um so flüchtiger sind.
Und was noch mehr ist, mein Sohn: -- besitzt ein enthusiastischer,
feuriger, ehrgeiziger, junger Mann eine Gattin, auf deren Namen ein
Flecken haftet, der, obwohl nicht ihre Schuld, von kalten und gemein
denkenden Leuten ihr und vielleicht auch ihren Kindern, und zwar um so
mehr zum Vorwurf gemacht wird -- um deswillen sie wie er um so mehr
Spott und Hohn zu erdulden haben -- je erfolgreicher oder glänzender
seine Laufbahn ist, so kann ihn -- und wenn er noch so gut und edel
ist -- im späteren Leben die Verbindung reuen, die er in seiner Jugend
geschlossen, und sie selbst den Schmerz und die Pein erfahren, es zu
wissen.»

«Mutter,» entgegnete der junge Mann ungeduldig, «ein solcher Mann wäre
ein elender Egoist, unwürdig des Namens eines Mannes und einer Frau,
wie Sie sie geschildert haben.»

«So denkst du jetzt, Harry!»

«Und ich werde stets so denken! Die Herzensqual, die ich in den
beiden letzten Tagen erduldet, dringt mir das offene Geständnis einer
Leidenschaft ab, die, wie Ihnen wohl bekannt, weder von gestern, noch
eine jugendlich-leichtsinnige und unbedachte ist. Meine Neigung zu dem
lieben, herrlichen Mädchen ist so tief und fest begründet, wie es die
Neigung eines Mannes nur sein kann. Ich habe keinen Gedanken, keinen
Lebensplan, keine Hoffnung außer ihr, höher als sie, und wenn Sie sich
meiner Liebe zu ihr widersetzen, so vernichten Sie meine Ruhe, mein
ganzes Glück für immer. O Mutter, überlegen Sie noch einmal und denken
Sie besser von mir; mißachten Sie die heißen Gefühle nicht, auf welche
Sie einen so geringen Wert zu legen scheinen.»

«Harry,» entgegnete Mrs. Maylie, «ich halte vielmehr so viel von warmen
und gefühlvollen Herzen, daß ich ihnen eine Enttäuschung ersparen
möchte. Doch wir haben für jetzt genug und mehr als genug von der Sache
geredet.»

«So überlassen Sie Rose die Entscheidung; und Sie werden sicher Ihren
zu strengen Ansichten nicht so viel Macht einräumen, daß Sie mir
Hindernisse in den Weg legen.»

«Das nicht; allein ich wünsche, daß du wohl überlegst --»

«Ich habe überlegt -- jahrelang überlegt -- fast so lange, wie ich mit
Ernst zu überlegen fähig bin. Meine Gefühle sind unverändert geblieben
-- werden stets unverändert bleiben, und warum sollte ich die Pein des
Aufschiebens und Wartens erdulden, was ja schlechterdings keinen Nutzen
haben kann. Ja, Rose muß mich anhören, bevor ich wieder abreise!»

«Sie soll es», sagte Mrs. Maylie.

«Ihr Ton scheint fast anzudeuten, daß sie mich kalt anhören wird,
Mutter», sagte der junge Mann angstvoll.

«Nichts weniger als dies», erwiderte die alte Dame; «weit entfernt
davon.»

«Hat sie auch wirklich keine andere Neigung?»

«Nein; ich müßte sehr irren, wenn du ihr Herz nicht bereits in nur zu
hohem Maße besäßest. -- Höre mich an,» fuhr sie fort, als ihr Sohn im
Begriff stand, zu antworten; «ich will nur noch dieses sagen. Bedenke,
ehe du dein Alles auf diesen Wurf setzest, ehe du dich zur höchsten
Hoffnungsstufe emportragen lässest, bedenke Roses Lebensgeschichte,
mein lieber Sohn, und überlege, welche Wirkung es auf ihre Entscheidung
haben kann, wenn sie von ihrer zweifelhaften Herkunft in Kenntnis
gesetzt wird; -- denn sie ist uns mit aller Innigkeit ihres edlen
Gemüts ergeben, und die vollkommenste Selbstaufopferung in großen wie
in geringen Dingen bezeichnete stets ihre Denkart.»

«Was wollen Sie damit sagen, Mutter?» fragte der junge Mann.

«Ich will es dir zu erraten überlassen», versetzte Mrs. Maylie. «Ich
muß wieder zu Rose gehen. Gott sei mit dir!»

«Werde ich Sie heute abend noch wiedersehen?»

«Ja, sobald ich Rose verlasse.»

«Werden Sie ihr sagen, daß ich hier bin?»

«Natürlich.»

«Und auch, welche Herzensangst ich um ihretwillen ausgestanden und wie
mich verlangt, sie zu sehen? Sie werden mir diesen Liebesdienst nicht
verweigern?»

«Nein, auch das will ich ihr sagen», erwiderte Mrs. Maylie, drückte dem
Sohne zärtlich die Hand und ging.

Losberne und Oliver hatten während dieser flüchtigen Unterredung am
fernsten Ende des Zimmers geweilt. Der erstere begrüßte jetzt Harry
Maylie auf das herzlichste und mußte ihm sofort den umständlichsten
Bericht über die Krankheit und das Befinden der Patientin erstatten.
Giles hörte mit begierigem Ohre zu, während er mit dem Gepäck
beschäftigt war.

«Haben Sie kürzlich etwas Besonderes geschossen, Giles?» fragte der
Doktor nach dem Schlusse seiner Mitteilungen.

«Nein, Sir, Besonderes eben nicht», erwiderte Giles, hoch errötend.

«Auch keine Diebe gefangen oder Räuber ausfindig gemacht?» fuhr
Losberne ein wenig boshaft fort.

«Nein, Sir», antwortete Giles sehr ernst.

«Das tut mir leid, da Sie sich auf dergleichen so vortrefflich
verstehen. Wie geht es denn Brittles?»

«Der junge Mensch befindet sich sehr wohl, und läßt sich Ihnen ganz
gehorsamst empfehlen, Sir.»

«Schön», sagte der Doktor. «Doch da ich Sie hier treffe, fällt mir's
ein, Giles, daß ich in den Tagen, wo ich so eilig abgerufen wurde,
aufgefordert von Ihrer gütigen Herrschaft, einen kleinen Auftrag zu
Ihren Gunsten übernahm. Treten Sie doch auf einen Augenblick mit mir an
das Fenster!»

Giles trat ziemlich verwundert zu ihm, und der Doktor beehrte ihn mit
einer kurzen, heimlichen Unterredung, nach deren Beendigung er eine
große Menge Verbeugungen machte, und mit ungewöhnlicher Wichtigkeit
wieder zurückging. Der Gegenstand des so leise geführten Gesprächs
wurde im Zimmer nicht bekannt gegeben, wohl aber sofort in der
Küche; denn dahin lenkte Mr. Giles augenblicklich seine Schritte und
verkündete, nachdem er sich einen Krug Ale hatte reichen lassen, daß
es seiner Herrschaft, in Anbetracht seines mutvollen Benehmens bei
dem Einbruche, gefallen habe, die Summe von fünfundzwanzig Pfund in
der Sparkasse für ihn niederzulegen. Die Köchin und das Hausmädchen
hoben die Hände und Augen empor und meinten, daß Mr. Giles jetzt ganz
stolz werden würde, worauf Mr. Giles, an seiner Hemdkrause zupfend,
erwiderte, daß sie sich in einem großen Irrtume befänden, und daß er
ihnen dankbar sein wollte, wenn sie, falls sie dergleichen jemals
gewahrten, ihn aufmerksam darauf machen würden, daß er sich hoffärtig
gegen Geringere erwiese. Er verbreitete sich darauf weitläufig über
seine Bescheidenheit und Anspruchslosigkeit, wofür ihm großes Lob
gezollt wurde, wie es bei bedeutenden Personen in solchen Fällen zu
geschehen pflegt.

Oben verging der Rest des Abends sehr heiter, denn der Doktor befand
sich in der fröhlichsten Stimmung, und so ermüdet oder nachdenklich
Harry Maylie anfangs gewesen sein mochte, er konnte der guten Laune des
wackeren Mannes nicht widerstehen. Losberne scherzte und erzählte, und
Oliver glaubte nie in seinem Leben so drollige Dinge gehört zu haben,
so daß er zur großen Freude des Doktors fortwährend lachte, wie der
Doktor selbst, und endlich auch Harry; denn auch das Gelächter hat ja
seine ansteckende Kraft. Mit einem Wort, sie waren so vergnügt, wie sie
es unter den obwaltenden Umständen nur irgend hätten sein können, und
es war spät geworden, als sie mit leichtem und dankerfülltem Herzen die
Ruhe aufsuchten, deren sie nach der Ungewißheit und Angst, in der sie
in den letzten Tagen geschwebt hatten, so sehr bedurften.

Oliver ging am folgenden Morgen mit mehr Hoffnung und Freude, als er,
wie ihm schien, seit langer Zeit gekannt hatte, an seine gewöhnliche
Beschäftigung. Die Betrübnis war von seinem Antlitz wie durch Zauber
verschwunden; es war ihm, als wenn die Blumen mit doppeltem Glanze im
Tau funkelten, die linde Luft in den Blättern lieblicher säuselte,
der Himmel reiner und blauer als je wäre. Das ist die Wirkung unserer
inneren Stimmung auf unsere Anschauung des Äußeren um uns her. Die auf
die Natur und ihre Mitmenschen blicken und wehklagen, daß alles schwarz
und finster sei, sie haben recht; allein die düsteren Farben sind
Widerspiegelungen ihrer gelbsüchtigen Augen und Herzen. Die wahren und
wirklichen sind zarte Tinten, und bedürfen eines schärferen Gesichts.

Eine bemerkenswerte Beobachtung entging auch Oliver nicht, nämlich, daß
er seine Morgenausflüge nicht mehr allein zu machen brauchte. Nachdem
ihn Harry Maylie zum erstenmal mit einer Blumenladung hatte heimkehren
sehen, wurde er von einer solchen Leidenschaft für Blumen ergriffen,
und er entwickelte so viel Geschmack im Ordnen derselben, daß er Oliver
weit hinter sich zurückließ, der dagegen wußte, wo die schönsten
Blumen zu finden waren. Sie durchstreiften Tag für Tag die Umgegend
miteinander, und brachten die köstlichsten Sträuße mit nach Hause.
Roses Fenster wurde jetzt geöffnet, denn die balsamische Sommerluft
erquickte sie, und auf der Fensterbank stand jeden Morgen ein frischer,
mit großer Sorgfalt geordneter Blumenstrauß. Oliver bemerkte, daß die
welken Blumen nie weggeworfen wurden, und daß der Doktor, wenn er durch
den Garten ging, stets hinaufblickte und bedeutsam lächelnd den Kopf
hin und her wiegte. So verflossen die Tage, und Roses Herstellung ging
rasch und glücklich vonstatten.

Auch unserm Oliver verging die Zeit nicht langsam, obwohl die junge
Dame ihr Zimmer noch nicht verlassen hatte, und obwohl es keine
Spaziergänge wie sonst mehr gab, ausgenommen dann und wann ganz kurze
mit Mrs. Maylie. Er verdoppelte seinen Fleiß in den Lehrstunden des
silberhaarigen, alten Mannes, so daß ihn seine raschen Fortschritte
fast selber wundernahmen. Eines Abends, als er seine Aufgaben für
den folgenden Tag lernte, begegnete ihm ein so unerwarteter wie als
Besorgnis erregender Vorfall.

Das kleine Zimmer, in welchem er bei seinen Büchern zu sitzen pflegte,
befand sich im Erdgeschoß, und lag nach hinten hinaus. Das Fenster ging
in den Garten, aus welchem man durch eine Tür auf einen eingehegten
Wiesengrund gelangte, und aus diesem auf den Anger und in ein Gehölz.
Es fing an zu dämmern, Oliver hatte fleißig gelesen und auswendig
gelernt, es war noch immer sehr warm, auch wohl ein wenig schwül, und
er schlummerte über einem Buche ein.

Uns beschleicht bisweilen eine Art von Schlummer, der, während er den
Leib gefangen hält, der Seele ein Halbbewußtsein der Umgebung und
die Fähigkeit, nach Belieben umherzuschweifen, läßt. Er ist Schlaf,
sofern eine überwältigende Schwere, eine Lähmung der Willenskraft
und eine gänzliche Unfähigkeit, unsere Gedanken und Vorstellungen zu
beherrschen, Schlaf genannt werden kann; dennoch aber wissen wir in
diesem Zustande, auch wenn wir träumen, was um uns her vorgeht, schauen
es, hören, was gesprochen wird, oder welche wirkliche Laute sonst an
unser Ohr dringen mögen, und Wirklichkeit und Einbildung vermischen
sich endlich so wunderbar, daß es nachgehends fast unmöglich ist,
sie wieder voneinander zu trennen. Es ist Tatsache, obwohl unsere
Gefühls- und Gesichtsorgane für die Zeit gleichsam tot sind, daß die im
Schlummer uns kommenden Gedanken und die in der Einbildung geschauten
Dinge bestimmt, und zwar wesentlich bestimmt werden durch die bloße
stumme Gegenwart eines wirklichen Gegenstandes, der uns, als wir die
Augen schlossen, nicht nahe zu sein brauchte, und von dessen Herannahen
oder Anwesenheit wir kein eigentliches Bewußtsein haben.

Oliver wußte genau, daß er sich in seinem kleinen Zimmer befand, daß
seine Bücher vor ihm auf dem Tische lagen, und daß der Abendwind in
dem Blätterwerk vor dem Fenster rauschte -- und schlummerte dennoch.
Plötzlich trat eine gänzliche Umwandlung seiner Umgebung ein, die Luft
wurde heiß und drückend, und er glaubte sich unter Angst und Schrecken
wieder im Hause des Juden zu befinden. Da saß der fürchterliche, alte
Mann in dem Winkel, in welchem er zu sitzen pflegte, wies mit dem
Finger nach ihm und flüsterte einem anderen, neben ihm sitzenden Manne,
der das Gesicht abgewendet hatte, etwas zu.

«Pst! mein Lieber!» glaubte er den Juden sagen zu hören; «er ist's,
ist's ohne Zweifel. Kommt -- laßt uns gehen!»

«Meint Ihr, daß ich ihn nicht erkannte?» schien der andere zu
antworten. «Und wenn eine Rotte von Teufeln seine Gestalt annähme, und
er stände mitten zwischen ihnen, so würd's mir mein Sinn zutragen,
welcher er wäre, und ich fände ihn heraus. Wenn Ihr ihn fünfzig Schuh
tief begrübet und brächtet mich über sein Grab, so würd' ich wissen,
und wenn auch kein Merkmal oder Zeichen es andeutete, daß er darunter
begraben läge. Möge sein Fleisch und Bein verfaulen, ich würd's!»

Der Mann schien die Worte in einem so tödlichen Haß verkündenden Tone
zu sprechen, daß Oliver bebend aufschreckte.

Gütiger Himmel, welcher Anblick war es, der ihm das stockende Blut zum
Herzen zurücktrieb und ihn der Stimme wie der Bewegungskraft beraubte!
Dort -- dort am Fenster -- nur zwei Schritte von ihm entfernt -- so
nahe, daß er ihn fast hätte berühren können, ehe er zurückschreckte --
stand, in das Zimmer hereinlugend, der Jude, dessen Blicke den seinigen
begegneten, und neben ihm gewahrte Oliver denselben Mann, der ihm vor
einiger Zeit im Hofe des Gasthauses ein solches Entsetzen eingejagt;
und der Fürchterliche war blaß vor Wut oder Grauen oder welcher inneren
Bewegung sonst, und seine Augen schossen drohende, zornige Blicke nach
Oliver!

Doch sie standen da, und Oliver sah sie nur einen einzigen, flüchtigen
Augenblick: dann waren sie verschwunden. Sie hatten indes ihn und
er hatte sie erkannt, und ihr Hereinlugen nach ihm und ihre Mienen
drückten sich seinem Gedächtnis so fest und tief ein, als wenn sie in
Stein ausgehauen und ihm von Kindheit an stets vor Augen gewesen wären.
Er stand einen Augenblick wie angewurzelt da, sprang darauf aus dem
Fenster in den Garten, und rief laut nach Hilfe.




35. Kapitel.

    Das Endergebnis des Abenteuers, das Oliver begegnet war, und eine
    Unterredung von ziemlicher Wichtigkeit zwischen Harry Maylie und
    Rose.


Als die Bewohner des Hauses, veranlaßt durch Olivers Rufen, in den
Garten eilten, fanden sie ihn bleich und bebend dastehen. Er wies nach
dem Wiesengrunde hinter dem Garten und war kaum imstande, die Worte zu
stammeln: «Der Jude! der Jude!»

Mr. Giles vermochte gar nicht zu fassen, was sie bedeuten sollten;
Harry Maylie, der Olivers Geschichte von seiner Mutter gehört hatte,
begriff es dagegen desto rascher.

«Welche Richtung hat er genommen?» fragte er, zugleich einen tüchtigen
Stock aufhebend, der zufällig dalag.

Oliver wies nach der Richtung hin, in welcher er die beiden Männer
hatte forteilen sehen und sagte, daß er sie soeben erst aus den Augen
verloren hätte.

«Dann wollen wir sie schon wieder einholen!» sagte Harry. «Folgt mir,
und haltet euch mir so nahe, wie ihr könnt!»

Er sprang bei diesen Worten über die Hecke und eilte so raschen
Laufes davon, daß die anderen ihm kaum zu folgen vermochten. Nach ein
paar Minuten gesellte sich ihnen auch Losberne, der eben von einem
Spaziergange heimkehrte, zu und rief ihnen laut die Frage zu, was denn
vorgefallen sei. Sie hielten erst an, um Atem zu schöpfen, als Harry in
das Angerstück einlenkte, nach welchem Oliver hingewiesen hatte, und
sorgfältig den Graben und die Hecke zu durchsuchen anfing, wodurch die
übrigen Zeit gewannen, heranzukommen und Losberne die Veranlassung der
Jagd mitzuteilen.

Ihr Suchen war vergeblich. Sie entdeckten nicht einmal frische
Fußspuren. Sie standen endlich auf einem kleinen Hügel, von welchem
aus sie die Wiesen, Anger und Felder nach allen Richtungen weithin
übersehen konnten. Linker Hand lag das kleine Dorf; allein die
Verfolgten hätten, um es zu erreichen, in der von Oliver beschriebenen
Richtung eine Strecke über den offenen Anger zurücklegen müssen, die
sie in so kurzer Zeit zurückzulegen schlechterdings nicht imstande
gewesen wären. Nach einer anderen Seite begrenzte dichtes Gebüsch
die Wiesen, allein es war aus dem gleichen Grunde unmöglich, daß sie
dasselbe schon hatten gewinnen können.

«Du mußt geträumt haben, Oliver», sagte Harry Maylie, ihn beiseite
führend.

«Nein, nein, Sir, wahrlich nicht», erwiderte der Knabe schaudernd; «ich
sah ihn zu deutlich -- sah beide so deutlich, wie ich Sie jetzt vor mir
sehe.»

«Wer war denn der andere?» fragten Harry und Losberne zugleich.

«Derselbe Mann, von dem ich Ihnen sagte, daß ich ihn im Hofe des
Gasthauses getroffen», antwortete Oliver. «Wir hatten unsere Blicke
wechselseitig aufeinander geheftet, und ich könnte es beschwören, daß
er es war.»

«Weißt du gewiß, daß sie diesen Weg genommen haben?» fragte Maylie.

«So gewiß, wie ich weiß, daß sie vor dem Fenster standen», versicherte
Oliver, und wies nach der Hecke zwischen dem Garten und dem
Wiesengrunde hinunter. «Da sprang der große Mann hinüber; der Jude lief
einige Schritte weit rechts und drängte sich durch die Lücke dort.»

Maylie und Losberne sahen Oliver und sodann einander an -- und man
brauchte nur die eifrigen Mienen des Knaben zu beobachten, um überzeugt
zu sein, daß er die reine Wahrheit sagte. Indes waren immer noch
keinerlei Spuren von Männern, die auf eiliger Flucht begriffen gewesen
wären, in irgendwelcher Richtung zu entdecken. Das Gras war lang, aber
nur da niedergetreten, wo die Verfolgenden es niedergetreten hatten.
Die Ränder und Seiten der Gräben waren von feuchter Tonerde, allein
an keiner Stelle wollte sich auch nur die mindeste Spur frischer
Fußstapfen finden.

«Es ist höchst auffallend», sagte Maylie.

«Höchst auffallend», wiederholte Losberne. «Sogar Blathers und Duff
würde der Verstand dabei stillstehen.»

Sie suchten noch immerfort, bis es vollkommen dunkel geworden war,
und sahen sich endlich genötigt, ihre Bemühungen ohne alle Hoffnung
auf Erfolg aufzugeben. Giles mußte sich die beiden ominösen Männer so
gut wie möglich von Oliver beschreiben lassen und wurde darauf in die
Bierhäuser des Dorfes abgeschickt, um Nachfragen anzustellen; er kehrte
jedoch zurück, ohne die mindeste Auskunft erhalten zu haben, indem
man sich doch zum wenigsten des Juden sicher erinnert haben würde,
wenn er verweilt, sich etwa einen Trunk reichen lassen oder mit jemand
gesprochen hätte.

Am folgenden Morgen wurden die Nachsuchungen und Nachforschungen
wiederholt, allein ebenso vergeblich. Am zweiten Tage ging Mr. Maylie
mit Oliver nach dem Marktflecken, in der Hoffnung, dort etwas von
dem Juden und seinem Begleiter zu sehen, zu hören oder zu erfahren;
doch der Versuch zeigte sich nicht minder fruchtlos als alle ihm
vorhergegangenen, und nach Verlauf einiger Tage fing die Sache an in
Vergessenheit zu geraten.

Rose hatte inzwischen das Krankenzimmer verlassen, konnte wieder
ausgehen, war dem Familienkreise zurückgegeben und erfreute aller
Herzen durch ihr Aussehen wie durch ihre Gegenwart.

Allein obgleich diese glückliche Veränderung die sichtbarste Wirkung
auf den kleinen Kreis hatte und obgleich in Mrs. Maylies Landhäuschen
wieder muntere Gespräche und fröhliches Gelächter gehört wurden, so
herrschte doch bisweilen eine sonst nicht gewöhnliche Zurückhaltung,
was auch Oliver nicht entging. Mrs. Maylie und ihr Sohn entfernten
sich oft und lange, und auf Roses Wangen waren Spuren von Tränen
bemerkbar. Nachdem der Doktor einen Tag zu seiner Abreise nach Chertsey
bestimmt hatte, lag es klar vor Augen, daß etwas vorging, wodurch der
Seelenfriede der jungen Dame und noch jemandes gestört wurde.

Als endlich Rose eines Morgens im Wohnzimmer allein war, trat Harry
Maylie herein und bat mit einigem Stocken um die Erlaubnis, ein paar
Worte mit ihr reden zu dürfen.

«Wenige, sehr wenige werden hinreichen, Rose», sagte der junge Mann,
sich zu ihr setzend. «Was ich dir zu sagen habe, ist dir bereits nicht
mehr unbekannt; du kennst die süßesten Hoffnungen meines Herzens,
obgleich du sie noch niemals aus meinem Munde vernommen hast.»

Rose war bei seinem Eintreten erblaßt, was freilich noch als eine
Nachwirkung ihrer Krankheit gedeutet werden konnte. Sie beugte sich
über einen ihr nahestehenden Blumentopf und wartete schweigend, daß er
fortfahren würde.

«Ich -- ich hätte schon früher wieder abreisen sollen», sagte er.

«Ich bin deiner Meinung, Harry», erwiderte Rose. «Vergib mir, daß ich
es sage, allein ich wollte, du hättest es getan.»

«Die schrecklichsten und quälendsten aller Befürchtungen haben mich
hergetrieben», entgegnete der junge Mann; «die Angst und Sorge,
das teure Wesen zu verlieren, auf das sich alle meine Wünsche und
Hoffnungen beziehen. Du warst dem Tode nahe -- standest bebend zwischen
Himmel und Erde. Wenn die Jugendlichen, Schönen und Guten durch
Siechtum heimgesucht werden, so wenden sich ihre reinen Geister den
ewigen Wohnungen seliger Ruhe zu, und deshalb sinken die Besten und
Schönsten unseres Geschlechts so oft in der Blüte ihrer Jugend in das
Grab.»

Der holden Jungfrau traten, als sie diese Worte vernahm, Tränen in die
Augen, und als eine derselben auf die Blume herabträufelte, über welche
sie sich niedergebeugt hatte, und diese verschönend hell in ihrem
Kelche glänzte, da war es, als wenn die Ergüsse eines reinen jungen
Herzens ihre Verwandtschaft mit den lieblichsten Kindern der Natur
geltend machten.

«Ein Engel,» fuhr der junge Mann leidenschaftlich fort, «ein Wesen,
so schön und frei von Schuld, wie ein Engel Gottes, schwebte zwischen
Leben und Tod. Oh, wer konnte hoffen, daß sie zu den Leiden und Ängsten
dieser Welt zurückkehren würde, als die ferne, ihr verwandte ihrem
Blicke schon halb geöffnet war! Rose, Rose! es war fast zu viel, um
es tragen zu können, zu wissen, daß du gleich einem leisen Schatten,
den ein Licht vom Himmel auf die Erde wirft, entschwändest -- keine
Hoffnung zu haben, daß du denen erhalten würdest, die hier noch
weilen, und keinen Grund zu kennen, warum du es solltest -- zu wissen,
daß du der schöneren Welt angehörtest, wohin so viele Reichbegabte in
der Kindheit und Jugend den zeitigen Flug gerichtet -- und doch bei all
solchen Tröstungen zu flehen, daß du den dich Liebenden wiedergegeben
werden möchtest! Das waren meine Gedanken bei Tag und Nacht, und mit
ihnen ergriff mich ein so überwältigender Strom von Besorgnissen und
Ängsten und selbstsüchtigen Schmerzen, daß du sterben und nie erfahren
würdest, wie heiß ich dich liebte, daß er mir in seinen Strudeln Sinn
und Verstand fast mit fortriß. Du genasest -- Tag für Tag und fast
Stunde für Stunde träufelten wieder Tropfen der Gesundheit aus Hygieias
Kelche herab und vermischten sich mit dem schwachen, fast versiegten,
zögernd in dir umlaufenden Lebensbächlein und schwellten es wieder
zum vollen, raschen, munteren Hinrieseln an. Ich habe dich mit Augen,
feucht vom heißesten Sehnen und innerster tiefer Herzensneigung,
zurückkehren sehen vom Tode zum Leben. Oh, sag' mir nicht, du
wünschtest, daß ich meine Liebe aufgegeben haben möchte, denn sie hat
mein Herz erweicht und der ganzen Menschheit geöffnet!»

«Das wollte ich nicht sagen», nahm Rose weinend das Wort; «ich wünsche
nur, daß du von hier fortgegangen sein möchtest, um dich wieder hohen
und edeln Bestrebungen -- deiner würdigen Bestrebungen zu widmen.»

«Es gibt keine Bestrebung, die meiner würdiger -- des edelsten und
herrlichsten Geistes würdiger wäre als das Mühen, ein Herz wie das
deinige zu gewinnen», versetzte der junge Mann, ihre Hand ergreifend.
«Rose, meine liebe, unnennbar teure Rose, ich habe dich seit -- ja,
seit Jahren geliebt, jugendlich hoffend und träumend, mein Teilchen
Ruhm mir zu erringen und dann stolz heimzukehren und im selben schönen
Augenblick dir zu sagen, daß ich das Errungene nur gesucht, um es mit
dir zu teilen, dich zu erinnern an die vielen stummen Zeichen einer
Jünglingsneigung, die ich dir gegeben, dir dein Erröten dabei in das
Gedächtnis zurückzurufen und dann deine Hand wie zur Besiegelung
eines unter uns altbestandenen, stillschweigenden Vertrags zu fordern.
Die Zeit ist noch nicht gekommen; doch gebe ich dir jetzt, ohne Ruhm
geerntet, ohne einen der jugendlichen Träume erfüllt gesehen zu haben,
das so lange schon dein gewesene Herz und setze mein Alles auf die
Erwiderung, die meiner Anerbietung von dir zuteil wird.»

«Deine Handlungsweise war immer gut und edel», erwiderte Rose, ihre
heftige Bewegung unterdrückend. «Glaubst du, daß ich weder fühllos noch
undankbar bin, so höre meine Antwort.»

«Geht sie dahin, daß ich mich bemühen soll, dich zu verdienen, teuerste
Rose?»

«Dahin, daß du dich bemühen mußt, mich zu vergessen -- nicht als deine
alte, liebe Gespielin, denn das würde mich unsäglich tief verwunden und
schmerzen, sondern als einen Gegenstand deiner Liebe. Blick' hinaus in
die Welt -- oh, wie viele Herzen gibt es in ihr, die du gleich stolz
sein kannst zu gewinnen. Vertraue mir eine Leidenschaft für eine andere
an, und ich will dir die wahrhafteste, wärmste und treueste Freundin
sein.»

Beide schwiegen, und Rose verhüllte ihr Antlitz und ließ ihren Tränen
freien Lauf. Harry hielt noch immer ihre Hand stumm in der seinigen.
«Und deine Gründe, Rose», begann er endlich mit leiser Stimme; «darf
ich die Gründe wissen, die dich zu dieser Entscheidung drängen?»

«Du hast ein Recht, nach ihnen zu fragen,» erwiderte Rose, «kannst
indes nichts sagen, was meinen Beschluß zu ändern vermöchte. Es ist
eine Pflicht, die ich üben muß. Ich bin es andern schuldig wie mir
selbst.»

«Dir selbst?»

«Ja, Harry, ich bin es mir selber schuldig, daß ich, ein verwaistes,
vermögensloses Mädchen mit einem Flecken auf meinem Namen, der Welt
keinen Grund gebe, zu wähnen, ich hätte aus niedrigen Antrieben
deiner ersten Leidenschaft nachgegeben und mich als ein Bleigewicht
an deine Hoffnungen und Entwürfe geheftet. Ich bin es dir und deinen
Angehörigen schuldig, dir zu wehren, im Feuer deiner edlen Gefühle ein
solches Hemmnis deines Vorwärtsschreitens in der Welt dir aufzubürden.»

«Wenn deine Neigungen mit deinem Pflichtgefühl zusammenstimmen --»,
begann Harry.

«Das ist nicht der Fall», unterbrach ihn Rose, tief errötend.

«So erwiderst du also meine Liebe?» sagte Harry. «Sage mir nur dies
eine, nur dies eine, Rose, und lindere die Bitterkeit meiner harten
Täuschung.»

«Wenn ich dürfte, ohne ihm, den ich liebte, ein schweres Leid
zuzufügen,» erwiderte Rose, «so würde ich --»

«So würdest du die Erklärung meiner Liebe ganz anders aufgenommen
haben?» fiel Harry in der größten Spannung ein. «O Rose, verhehle mir
das wenigstens nicht.»

«Nun ja», sagte die Jungfrau. «Doch», fügte sie, ihre Hand der seinigen
entziehend, hinzu, «warum diese peinliche Unterredung fortsetzen, die
für mich am schmerzlichsten, wenn auch ein Quell der reinsten Freude
ist? Denn es wird mir allerdings stets ein hohes Glück gewähren, einst
von dir wie jetzt beachtet und geliebt zu sein, und jeder neue Triumph,
den du im Leben erringst, wird mich mit neuer Kraft und Festigkeit
erfüllen. Lebe wohl, Harry, denn wir dürfen uns so nie wiedersehen,
wenn uns auch in anderen Beziehungen die schönsten, innigsten Bande
umschlingen. Möge dir jeder Segen zuteil werden, den das Flehen eines
treuen und aufrichtigen Herzens von dort, wo die Wahrheit thront und
alles Wahrheit ist, auf dich herabrufen kann!»

«Noch ein Wort, Rose», sagte Harry. «Deine wahren, eigentlichen Gründe.
Laß sie mich aus deinem eigenen Munde hören.»

«Deine Aussichten sind glänzend», erwiderte sie mit Festigkeit. «Dir
winken alle Ehren, zu denen bedeutende Talente und einflußreiche
Verbindungen zu verhelfen vermögen. Aber deine Anverwandten und
Gönner sind stolz, und ich will mich ihnen weder aufdrängen, die
Mutter verachten, die mir das Leben gab, noch auf den Sohn der Frau,
die Mutterstelle an mir vertrat, Unehre bringen oder schuld an der
Vereitelung seiner Hoffnungen und Aussichten sein. Mit einem Worte,»
fuhr sie, sich abwendend, als wenn die Festigkeit sie verließe,
fort, «es klebt ein Makel an meinem Namen, wie ihn die Welt an den
Unschuldigen nun einmal heimsucht; er soll in kein fremdes Blut
übergehen, sondern der Vorwurf auf mir allein haften bleiben.»

«Noch ein Wort, teuerste Rose -- noch ein einziges Wort», rief Harry,
sich vor ihr niederwerfend. «Wäre ich minder -- minder glücklich, wie
es die Welt nennt -- wäre mir ein dunkles und stilles Los beschieden
gewesen -- wäre ich arm, krank, hilflos -- würdest du mich dann auch
zurückweisen, oder entspringen deine Bedenken aus meinen vermuteten
Aussichten auf Reichtümer und Ehren?»

«Dränge mich nicht zu einer Antwort auf diese Frage», versetzte Rose.
«Es kann und wird keine Veranlassung kommen, sie aufzuwerfen, und es
ist nicht recht, nicht freundlich von dir --»

«Wenn deine Antwort lautete, wie ich es fast zu hoffen wage,»
unterbrach Harry das bebende Mädchen, «so würde ein Wonnestrahl auf
meinen einsamen Weg fallen und den düsteren Pfad vor mir erhellen.
Wieviel kannst du durch die wenigen kurzen Worte für mich tun, der ich
dich über alles liebe! O Rose, bei meiner glühenden, unvergänglichen
Neigung -- bei allem, was ich für dich gelitten und nach deinem
Ausspruche leiden soll -- beantworte mir die eine Frage!»

«Nun wohl!» erwiderte sie; «wenn dir ein anderes Los beschieden gewesen
wäre -- wenn du immerhin ein wenig, doch nicht so hoch über mir
ständest, wenn ich dir bei beschränkten Verhältnissen eine Gehilfin
und Trösterin sein könnte, statt in glänzenden dich nur zu hindern, zu
hemmen und zu verdunkeln, so würde ich dir diese ganze Pein erspart
haben. Ich habe jetzt alle, alle Ursache, zufrieden und glücklich zu
sein, würde dann aber, ich bekenne es, Harry, mein Glück erhöht achten.»

Lebhafte Erinnerungen an alte, süße Hoffnungen, die sie als aufblühende
Jungfrau lange gehegt, drängten sich ihr bei diesem Geständnisse auf
und brachten Tränen mit, wie es alte Hoffnungen tun, wenn sie verwelkt
vor der Seele auftauchen; allein sie schafften ihrem gepreßten Herzen
Erleichterung.

«Ich kann meiner Schwäche nicht wehren, und sie bestärkt mich in meinem
Entschluß», fügte sie, dem Geliebten die Hand reichend, hinzu. «In
Wahrheit, Harry, ich muß dich verlassen.»

«So bitte ich um ein Versprechen», flehte er. «Laß mich noch ein
einziges Mal -- in einem Jahr oder vielleicht noch weit früher -- ein
letztes Mal über diesen Gegenstand zu dir reden.»

«Nicht um in mich zu dringen, daß ich meinen wohlüberlegten Entschluß
ändere, Harry; es würde vergeblich sein», erwiderte Rose mit einem
wehmütigen Lächeln.

«Nein,» versetzte er, «um dich ihn wiederholen zu hören, wenn du ihn
wiederholen willst. Ich will dir, was ich mein nennen mag, zu Füßen
legen und der Entscheidung, die du jetzt ausgesprochen, wenn du bei ihr
beharrst, auf keinerlei Weise entgegentreten.»

«Dann sei es so», sagte Rose. «Es ist nur noch eine Bitterkeit mehr,
und ich vermag sie später vielleicht besser zu ertragen.»

Sie reichte ihm noch einmal die Hand; allein er drückte sie an seine
Brust, küßte ihre schöne Stirn und eilte hinaus.




36. Kapitel.

    Abermals ein kurzes Kapitel, das an seiner Stelle als nicht eben
    sehr wichtig erscheinen mag, aber doch gelesen werden sollte,
    weil es das vorhergehende erörtert, und einen Schlüssel zum
    nachfolgenden darbietet.


«Sie sind also entschlossen, heute morgen mit mir abzureisen?» fragte
der Doktor, als sich Harry Maylie mit ihm und Oliver zum Frühstück
niedersetzte. «Sie ändern ja Ihre Entschlüsse mit jeder halben Stunde.»

«Ich hoffe, daß Sie bald anderer Meinung sein werden», entgegnete
Maylie, sich ohne ersichtlichen Grund verfärbend.

«Ich wünsche sehr, Ursache dazu zu bekommen,» versetzte Losberne,
«obgleich ich bekenne, daß ich daran zweifle. Gestern morgen hatten Sie
sehr eilfertig beschlossen, zu bleiben und als ein guter Sohn Ihre Frau
Mutter an die Seeküste zu begleiten; kurz vor Mittag erklärten Sie, daß
Sie mir die Ehre erweisen wollten, so weit mit mir zu fahren, wie ich
auf der Londoner Straße bliebe; und gegen Abend drangen Sie unsäglich
geheimnisvoll in mich, daß ich abreisen möchte, bevor die Damen
aufgestanden wären, wovon die Folge ist, daß Oliver hier beim Frühstück
festsitzt, während er botanisieren gehen sollte. Ist's nicht zu arg,
Oliver?»

«Es würde mich sehr betrübt haben, Sir, nicht zu Hause gewesen zu sein,
wenn Sie und Mr. Maylie abgereist wären», antwortete Oliver.

«Bist ein guter Junge,» sagte der Doktor, «sollst zu mir kommen, wenn
du zurückgekehrt bist. Doch um ernsthaft zu reden, Harry, hat eine
Mitteilung Ihrer hohen Gönner und Freunde Ihren Abreiseeifer bewirkt?»

«Ich habe,» erwiderte Maylie, «seit ich hier verweile, durchaus keine
Mitteilung von meinen Gönnern und Freunden, zu denen Sie ohne Zweifel
meinen Onkel zählen, erhalten, auch ist es nicht wahrscheinlich, daß
sich eben jetzt etwas ereignet, wodurch ich zu ihnen zu eilen mich
gedrungen fühlen könnte.»

«Sie sind ein schnurriger Kauz», fuhr der Doktor fort. «Indes werden
besagte Gönner Sie bei der Wahl vor Weihnachten natürlich ins Parlament
befördern, und Ihre plötzlichen Beschluß- und Willensänderungen sind
keine schlechte Vorbereitung auf das öffentliche Leben. Ein gutes
Trainieren ist allezeit wünschenswert, mag das Rennen Staatsstellen,
Ehrenbechern oder Rennpreisen gelten.»

Harry Maylie machte eine Miene, als wenn er den Doktor leicht genug
aus dem Felde schlagen könnte, begnügte sich indes zu sagen: «Wir
werden sehen», und ließ den Gegenstand fallen. Kurz darauf fuhr die
Postkutsche vor, Giles holte das Gepäck, und Losberne war eifrig
beschäftigt, die letzten Reisevorkehrungen zu beaufsichtigen.

«Ein Wort, Oliver», sagte Harry Maylie leise.

Oliver trat zu ihm in die Fenstervertiefung, in welcher er stand, sehr
verwundert über die stille Traurigkeit und Unruhe, die er zugleich an
ihm bemerkte.

«Du kannst jetzt recht gut schreiben», sagte Maylie, die Hand auf den
Arm des Knaben legend.

«Ziemlich», erwiderte Oliver.

«Ich komme vielleicht vorerst nicht wieder nach Hause und wünsche, daß
du mir schreibst, etwa einen Montag um den andern. Willst du?» fuhr
Harry fort.

«Mit Freuden, Sir!» rief Oliver äußerst erfreut über den Auftrag aus.

«Ich wünsche von dir zu hören, wie -- es meiner Mutter und Miß Maylie
ergeht; melde mir, was für Spaziergänge ihr macht, wovon ihr plaudert,
und ob sie sich wohl befinden und recht heiter sind. Du verstehst?»

«Vollkommen, Sir.»

«Auch wünsche ich, daß du ihnen nichts davon sagst; es möchte meine
Mutter beunruhigen, so daß sie sich bewogen fände, mir öfter zu
schreiben, was immer eine große Belästigung für sie ist. Also muß es
ein Geheimnis unter uns bleiben, und schreib mir ja alles; ich verlasse
mich auf dich.»

Oliver fühlte sich hochgeehrt, versprach, was von ihm verlangt wurde,
und Maylie sagte ihm unter vielen Versicherungen seiner Zuneigung
Lebewohl.

Der Doktor war bereits eingestiegen, die Dienerschaft wartete am Wagen,
Harry warf einen flüchtigen Blick nach Roses Fenster hinauf und stieg
gleichfalls ein.

«Fort, Postillon!» rief er, «und fahre, so schnell du kannst; ich werde
heute nur zufrieden sein, wenn es wie im Fluge geht.»

«Was fällt Ihnen ein?» rief der Doktor; «Postillon, ich werde nur
zufrieden sein, wenn es ganz und gar nicht im Fluge geht.»

Die Dienerschaft sah dem Wagen nach, solange er sichtbar war,
Rose aber, die hinter den Vorhängen gelauscht hatte, als Harry
hinaufblickte, schaute noch immer in die Ferne hinaus, als sich die
Dienerschaft schon längst wieder hineinbegeben hatte.

«Er scheint ganz heiter und zufrieden zu sein», sagte sie endlich.
«Ich fürchtete, daß das Gegenteil der Fall sein könnte, und freue mich
meines Irrtums.»

Tränen sind Zeichen sowohl der Freude wie des Schmerzes; die aber,
welche über Roses Wangen hinabträufelten, während sie sinnend und
fortwährend in derselben Richtung hinausschauend am Fenster saß,
schienen mehr Kummer als Lust zu bedeuten.




37. Kapitel.

    In welchem der Leser, wenn er in das sechsunddreißigste Kapitel
    zurückblicken will, einen im ehelichen Leben nicht selten
    hervortretenden Kontrast beobachten wird.


Mr. Bumble saß in seinem Wohnzimmer im Armenhause und blickte
nachdenklich und düster bald in den Kamin, in welchem kein Feuer
brannte, da es Sommer war, und der daher öde und trostlos genug aussah
und bald noch düsterer zu dem Leimzweige empor, der von der Decke
herabhing und von den ihr Verderben nicht ahnenden Fliegen umschwärmt
wurde. Vielleicht erinnerten ihn die Tierchen an eine traurige
Begebenheit seines eigenen Lebens.

Auch fehlte es nicht an sonstigen Anzeichen, daß in seinen
Angelegenheiten eine bedeutende Veränderung vorgegangen sein mußte.
Wo waren der Tressenrock und der dreieckige Hut? Er trug noch
Kniehosen und schwarze wollene Strümpfe -- doch es waren nicht die des
Kirchspieldieners. Der Rock war ein anderer. Der Hut ein gewöhnlicher,
bescheidener, runder. Mr. Bumble war nicht mehr Kirchspieldiener.

Es gibt Beförderungen im Leben, die, abgesehen von den mit ihnen
verknüpften materiellen Vorteilen, doch noch einen ganz besonderen Wert
und eine eigentümliche Würde durch das mit ihnen verknüpfte Kostüm
erhalten. Ein Feldmarschall hat seine Uniform, ein Bischof seinen
Ornat, ein Richter seine große Perücke, ein Kirchspieldiener seinen
dreieckigen Hut. Man nehme dem Richter seine Perücke, dem Bischof
seinen Ornat oder dem Kirchspieldiener seinen dreieckigen Hut, und was
sind sie? Weiter nichts mehr als Menschen -- bloße Menschen. Würde, und
bisweilen sogar Heiligkeit hängen mehr von Uniformen, Ornaten, Perücken
und Hüten ab, als viele Leute sich träumen lassen.

Mr. Bumble hatte Mrs. Corney geehelicht und war Armenhausverwalter. Ein
anderer Kirchspieldiener war zur Gewalt gelangt, und der dreieckige
Hut, der Tressenrock und der Stab waren auf ihn übergegangen.

«Morgen sind's zwei Monate!» sagte Mr. Bumble seufzend. «Es scheint ein
Jahrhundert zu sein.»

Mr. Bumble wollte vielleicht sagen, daß er in dem kurzen Zeitraum von
acht Wochen ein ganzes, glückliches Leben verlebt hätte -- allein der
Seufzer! Es lag gar viel in ihm.

«Ich verkaufte mich», fuhr Bumble fort, «für sechs Teelöffel, eine
Zuckerzange, einen Milchgießer, eine Stube voll alter Möbel und zwanzig
Pfund Geld -- nur gar zu billig, spottwohlfeil!»

«Wohlfeil!» tönte ihm eine schrille Stimme ins Ohr. «Du wärst für jeden
Preis zu teuer gewesen, und der Himmel weiß, daß ich dich mehr als zu
teuer bezahlt habe.»

Bumble drehte sich um und blickte in das Antlitz seiner liebenswürdigen
Ehehälfte, welche sein kurzes Selbstgespräch nur unvollkommen
verstanden und ihre erwähnte Bemerkung auf gut Glück hingeworfen hatte.

«Frau, sei so gut, mich anzusehen», sagte Bumble und dachte bei sich
selbst: «Wenn sie solch einen Blick aushält, so hält sie alles aus. Er
hat bei den Armen niemals seinen Zweck verfehlt, und verfehlt er ihn
bei ihr, so ist es mit meiner Macht und Gewalt vorbei.»

Er verfehlte seinen Zweck. Mrs. Bumble wurde keineswegs durch ihn
überwältigt, sondern erwiderte ihn durch einen äußerst verächtlichen,
und verband damit obendrein ein Gelächter, das zum wenigsten klang, als
wenn es ihr von Herzen käme.

Als Bumble die unerwarteten Töne vernahm, sah er zuerst ungläubig und
dann erstaunt aus, worauf er wieder in sein Brüten und Sinnen verfiel,
aus welchem ihn jedoch Mrs. Bumble erweckte. «Willst du den ganzen Tag
dasitzen und schnarchen?» fragte sie.

«Ich denke hier so lange sitzen zu bleiben, wie es mir beliebt»,
entgegnete er; «und obschon ich keineswegs schnarchte, so bin ich doch
gewillt, von meinem Rechte Gebrauch zu machen und ganz nach meinem
Gefallen zu schnarchen, zu niesen, zu lachen oder zu weinen, oder was
mir eben sonst behagt.»

«Von deinem Rechte!» höhnte Mrs. Bumble mit unsäglich verächtlicher
Miene.

«Ja, von meinem Rechte-1 Es ist das Recht des Mannes, nach seinem Willen
zu leben und zu befehlen.»

«Und was ist denn ins Kuckucks Namen das Recht der Frau?»

«Nach des Mannes Willen zu leben und zu gehorchen. Dein unglücklicher,
erster Mann hätte es dich lehren sollen; er wäre dann vielleicht noch
am Leben -- und ich wollte, daß er es wäre, der gute Mann!»

Mrs. Bumble erkannte, daß der entscheidende Augenblick gekommen war,
und daß es galt, sich der Herrschaft ein für allemal zu bemächtigen,
oder ihr für immer zu entsagen. Sie sank daher auf einen Stuhl nieder,
erklärte Mr. Bumble für einen Unmenschen mit einem Kieselherzen und
brach in einen Tränenstrom aus.

Allein Tränen waren es nicht, was zu Mr. Bumbles Herzen drang; es war
wasserdicht. Den Filzhüten gleich, welche gewaschen werden können und
durch Regen besser werden, wurden seine Nerven durch Tränenschauer
noch fester, die ihn als Zeichen der Schwäche und somit als
stillschweigende Anerkenntnisse seiner Obergewalt erfreuten und stolz
machten. Er blickte seine Hausfrau mit großer Zufriedenheit an und bat
und munterte sie auf alle Weise auf, nur immerzu zu weinen, und nach
besten Kräften, denn es sei äußerst gesund, wie die Ärzte versicherten.

«Es erweitert die Lungen, wäscht das Gesicht rein, schärft die Augen
und kühlt ein zu heißes Temperament ab», sagte er; «also weine ja nur
immerzu.» -- Nachdem er die scherzenden Worte gesprochen, griff er zu
seinem Hute, setzte ihn kecklich auf die eine Seite, wie ein Mann, der
seine Überlegenheit fühlt und auf geeignete Weise zeigen will, steckte
die Hände in die Taschen und setzte sich stolzierenden Schritts nach
der Tür in Bewegung.

Mrs. Bumble hatte einen Versuch mit den Tränen angestellt, weil sie
minder mühsam waren als ein Faustangriff; indes war sie vollkommen
bereit, eine Probe mit dem letzteren Verfahren zu machen, was Mr.
Bumble auch nicht lange verborgen blieb.

Die erste Kunde, welche er davon erhielt, bestand in einem dumpfen
Schalle, welcher die unmittelbare Folge hatte, daß sein Hut an das
äußerste Ende des Zimmers flog. Sobald durch dieses vorläufige Beginnen
sein Kopf entblößt war, packte ihn die erfahrene Dame mit der einen
Hand bei der Kehle und ließ mit der andern einen Hagel von Schlägen,
und zwar ebenso gewandt wie wirksam auf sein Haupt niederfallen.
Hierauf brachte sie ein wenig Abwechslung in ihr Vorgehen, indem
sie ihm das Gesicht zerkratzte und Hände voll Haare ausraufte, und
nachdem sie ihn nunmehr so nachdrücklich bestraft hatte, wie sie es
dem Vergehen nach für nötig erachtete, warf sie ihn über einen Stuhl,
der nicht zweckmäßiger hätte stehen können und forderte ihn auf, noch
einmal von seinen Rechten zu sprechen, wenn er es wagen wollte.

«Laß los!» rief er in befehlendem Tone, «und mach' sogleich, daß du
fortkommst, wenn du nicht willst, daß ich etwas Desperates tue.» Er
stand mit den allerkläglichsten Mienen auf, sann darüber nach, was
wohl ganz desperat sein möchte, hob seinen Hut auf und blickte nach der
Tür.

«Gehst du bald?» fragte Mrs. Bumble.

«Ich gehe schon, ja doch», erwiderte er, sich rasch nach der Tür
zurückziehend; «ich beabsichtige keineswegs -- wirklich, ich gehe
schon, Liebe -- du bist aber auch so heftig, daß ich fürwahr --»

Mrs. Bumble bückte sich in diesem Augenblick, um den in Unordnung
geratenen Teppich wieder zurecht zu schieben, und ihr Eheherr schoß
hinaus, ohne daran zu denken, seine Rede zu vollenden, und ließ weiland
Mrs. Corney im ungestörten Besitz des Schlachtfeldes. -- Mr. Bumble
war der Überraschung erlegen und ohne Frage vollständig in die Flucht
geschlagen. Er hatte die entschiedenste Neigung zum Bramarbasieren,
nichts konnte ihm größere Freude gewähren, als Verübung kleiner
Tyrannei und Grausamkeit, und er war demnach, wie kaum gesagt zu werden
braucht, eine Memme. Hierdurch wird indes sein Charakter keineswegs
heruntergesetzt, da so viele Beamte, die in hoher Achtung stehen und
höchlich bewundert werden, die Opfer ähnlicher Schwächen sind. Wir
haben jene Bemerkung vielmehr zu seinen Gunsten gemacht, und um unsern
Lesern noch mehr zu Gemüt zu führen, wie trefflich sich Bumble zu einem
Beamten eignete.

Das Maß seiner Erniedrigung war indes noch nicht voll. Nachdem er einen
Gang durch das ganze Haus gemacht und zum erstenmal daran gedacht
hatte, daß die Armengesetze doch wirklich zu streng wären, und daß
Männer, die von ihren Frauen fortliefen und die Erhaltung derselben dem
Kirchspiele aufbürdeten, von Rechts wegen ganz und gar nicht bestraft,
sondern vielmehr als verdiente Individuen und Märtyrer belohnt werden
sollten, kam er in ein Gemach, in welchem die Bewohnerinnen des
Armenhauses beschäftigt zu werden pflegten, das Kirchspielleinenzeug zu
waschen, und in welchem er lautes Sprechen hörte.

«Hm!» sagte er, seine ganze angeborene Würde annehmend; «zum wenigsten
sollen diese Weiber auch fernerhin meine Rechte achten. Holla -- Blitz
und Hagel! -- wie könnt ihr euch unterstehen, einen solchen Lärm zu
machen, verwünschtes Weibsvolk?»

Er öffnete mit diesen Worten die Tür, schritt hochfahrend und zornig
hinein, nahm jedoch unmittelbar darauf die demütigste Miene an, denn er
erblickte seine Hausehre. «Ich wußte nicht, daß du hier wärst, lieber
Schatz», sagte er.

«Wußtest nicht, daß ich hier war?» fuhr sie ihn an. «Was hast du denn
hier zu schaffen?»

«Ich dachte, sie sprächen zu viel, um ihre Arbeiten gehörig verrichten
zu können», erwiderte er, zerstreut nach ein paar alten Frauen an
einem Waschfasse hinblickend, die bewundernde Blicke ob der Demut des
Armenhausverwalters wechselten.

«Du dachtest, sie sprächen zu viel?» sagte Mrs. Bumble. «Was geht denn
dich das an?»

«Ei nun, lieber Schatz --»

«Ich frage noch einmal, was es dich angeht?»

«Es ist wahr, du hast hier zu befehlen, lieber Schatz; ich glaubte
aber, du wärest eben nicht bei der Hand.»

«Ich will dir was sagen, Bumble: wir brauchen dich hier nicht, du hast
hier nichts verloren und steckst deine Nase viel zu gern in Dinge,
die dich nichts angehen; machst dich bei jedermann lächerlich und zum
Narren und wirst ausgelacht, sobald du den Rücken wendest. Troll dich
-- willst du, oder willst du nicht?»

Bumble gewahrte mit folternden Gefühlen, wie die beiden alten
Wäscherinnen wahrhaft entzückt miteinander kicherten und zögerte
einen Augenblick. Mrs. Bumble, deren Geduld bei einem Aufschube nicht
Probe hielt, ergriff ein Gefäß mit Seifenwasser, näherte sich ihm und
wiederholte ihre Aufforderung, bei Strafe, im Falle des Ungehorsams,
seine stattliche Person überschüttet zu sehen.

Was konnte er tun? Er blickte trostlos umher, schlich nach der Tür,
und das Gekicher der Wäscherinnen verwandelte sich in ein schallendes
Gelächter. Mehr bedurfte es nicht. Er war in ihren Augen erniedrigt,
hatte Ehre und Ansehen sogar bei den Armen verloren, war von der Höhe
der Kirchspieldienerschaft zur tiefsten Tiefe des unter Weiberregiment
stehenden Ehemannes heruntergesunken. «Und das alles nach zwei
Monaten!» dachte Bumble. «Kaum vor zwei -- noch vor zwei kurzen Monaten
war ich mein eigener Herr und gebot über das ganze Armenhaus, und
jetzt!»

Es war zu viel. Er ohrfeigte den Knaben, der ihm das Tor öffnete (denn
er hatte mittlerweile das Portal erreicht) und trat zerstreut auf die
Straße.

Er ging eine Zeitlang auf und ab, bis sich die erste Heftigkeit seines
Kummers gelegt hatte. Sie ließ indes Durst zurück. Er schritt an vielen
Wirtshäusern vorüber und stand endlich vor einem in einem Nebengäßchen
befindlichen still, dessen Gaststube, wie er durch einen flüchtigen
Blick sich überzeugte, leer war. Nur ein einziger Mann saß darin. Es
fing eben an stark zu regnen, und dies bestimmte ihn. Er ging hinein
und forderte ein Glas Branntwein.

Der im Gastzimmer sitzende Mann war groß und schwärzlich und hatte
sich in einen weiten Mantel gehüllt. Er schien ein Fremder und
ziemlich weit gewandert zu sein, denn er sah ermüdet aus und hatte
staubige Stiefel an. Er blickte Bumble, als dieser eintrat, von der
Seite an, ließ sich aber zur Entgegnung seines Grußes kaum zu einem
Kopfnicken herab. Bumble besaß Würde genug für zwei, trank daher
sein Glas Branntwein mit Wasser stillschweigend, und nahm mit großer
Wichtigkeit ein Zeitungsblatt zur Hand. Wie es indes unter Umständen
dieser Art zu geschehen pflegt, er empfand eine starke Neigung, der
er nicht widerstehen konnte, von Zeit zu Zeit nach dem Unbekannten
verstohlen hinüberzublicken, worauf er stets die Augen etwas verwirrt
wieder niedersenkte, da der Unbekannte jedesmal dasselbe tat. Seine
Verwirrung wurde noch durch den auffallenden Ausdruck der Augen des
letzteren vergrößert, welche scharf und durchdringend waren, und aus
denen finstere, argwöhnische Blicke hervorschossen, wie Bumble sie noch
nie gesehen, und die seinen Mienen etwas höchst Abstoßendes gaben.

Als die Blicke beider einander auf diese Weise mehrmals begegnet waren,
brach endlich der Fremde das Stillschweigen.

«Sahen Sie nach mir,» hub er mit tiefer, rauher Stimme an, «als Sie in
das Fenster hereinblickten?»

«Nicht daß ich wüßte, sofern Sie nicht Mr--» Bumble unterbrach sich
hier selbst. Er wünschte den Namen des Fremden zu erfahren und hoffte,
daß derselbe sich nennen würde.

«Ah, Sie haben also nicht nach mir hereingeblickt,» sagte der
Unbekannte, spöttisch den Mund verziehend, «denn Sie würden sonst
meinen Namen kennen. Ich möchte Ihnen raten, nicht danach zu fragen.»

«Ich habe nichts Böses gegen Sie im Sinn, junger Mann», entgegnete
Bumble, sich in die Brust werfend.

«Und haben mir auch nichts Böses zugefügt», lautete die rasche Antwort.

Es trat wiederum Stillschweigen ein, das der Fremde nach einiger
Zeit zum zweitenmal unterbrach. «Ich sollte meinen, daß ich Sie
schon gesehen hätte. Sie waren zu der Zeit anders gekleidet, und ich
begegnete Ihnen nur auf der Straße, erkenne Sie aber wieder. Waren Sie
nicht Kirchspieldiener hier im Orte?»

Bumble bejahte nicht ohne einige Verwunderung.

«Was sind Sie denn jetzt?»

«Armenhausverwalter», erwiderte Bumble langsam und mit nachdrücklicher
Betonung, um den Unbekannten zu verhindern, einen Ton ungebührlicher
Vertraulichkeit anzunehmen. «Armenhausverwalter, junger Mann!»

«Sie werden sich doch ohne Zweifel noch ebensogut auf Ihren Vorteil
verstehen wie sonst?» fuhr der Unbekannte, ihn scharf anblickend,
fort, denn Bumble sah ihn nicht wenig erstaunt an. «Tragen Sie kein
Bedenken, mir offen zu antworten; Sie sehen ja, daß ich Sie genau genug
kenne.»

«Ein verheirateter Mann», versetzte Bumble, die Augen mit der Hand
beschattend und den Unbekannten in offenbarer Verlegenheit von
Kopf bis zu den Füßen betrachtend, «ist nicht abgeneigter als ein
alleinstehender, auf eine ehrliche Weise ein Stück Geld zu verdienen.
Die Kirchspielbeamten werden nicht so reichlich besoldet, daß sie eine
kleine Nebeneinnahme von der Hand weisen dürften, wenn sie sich ihnen
auf eine anständige und schickliche Weise darbietet.»

Der Unbekannte lächelte, nickte mit dem Kopfe, als wenn er sagen
wollte, daß er sich in seinem Manne nicht geirrt hätte, und klingelte.
Der Wirt erschien, er reichte ihm Bumbles leeres Glas und befahl ihm,
es mit starkem und heißem Getränk wieder zu füllen.

«Sie lieben es doch so?» sagte er.

«Nicht zu stark», erwiderte Bumble mit einem Zartgefühl ausdrückenden
Husten.

«Sie wissen schon, was das sagen will», rief der Unbekannte in
trockenem Tone dem Wirt nach, der lächelnd verschwand und kurz darauf
mit einem dampfenden Glase zurückkehrte, das Bumble das Wasser in die
Augen trieb.

«Hören Sie mich nun an», sagte der Unbekannte, sobald sie wieder allein
waren. «Ich bin heute hierher gekommen, um Sie aufzusuchen, und als
ich eben daran dachte, wie ich Sie treffen sollte, trieb Sie mir einer
der Zufälle in den Weg, durch die der Teufel bisweilen seine Freunde
zusammenführt. Ich muß eine Erkundigung bei Ihnen einziehen, und
verlange Ihre Mühe, so gering sie sein mag, nicht umsonst. Stecken Sie
das als Handgeld ein.»

Er legte ein paar Goldstücke vor ihn auf den Tisch, und nachdem Bumble
dieselben sorgfältig geprüft hatte, ob sie auch nicht falsch wären,
und sie vergnügt in die Tasche gesteckt hatte, fuhr der Fremde fort:
«Denken Sie einmal zurück -- ja an den Winter vor zwölf Jahren.»

«Das ist eine lange Zeit», sagte Bumble. «Aber schon gut. Ich denke an
den Winter.»

«Schauplatz das Armenhaus.»

«Gut.»

«Zeit die Nacht.»

«Ja, ja.»

«Ort das elende Loch, in welchem liederliche Weibsbilder Kindern das
ihnen selbst oft versagte Leben geben, Kindern, die das Kirchspiel
aufzuziehen hat, und wo sie sterbend ihre Schande verstecken.»

«Sie meinen das Wöchnerinnenzimmer», sagte Bumble.

«Ja. In ihm wurde ein Knabe geboren.»

«Viele, viele Knaben», erwiderte Bumble mit kläglichem Kopfschütteln.

«Hol' der Teufel die junge Höllenbrut!» rief der Unbekannte ungeduldig
aus. «Ich spreche von einem, 'nem zierlich und bläßlich aussehenden
Wichte, der bei einem Leichenbestatter in die Lehre getan wurde (ich
wollte, daß er selbst längst zu Grabe getragen wäre!) und später
fortlief, wie man glaubte, nach London.»

«Sie meinen Oliver -- den Oliver Twist? Ich erinnere mich seiner
natürlich sehr wohl. Wir hatten keinen eigensinnigeren kleinen
Schlingel im Hause --»

«Ich brauche nichts von ihm zu hören, habe genug von ihm gehört -- wo
ist die alte Hexe, die seine Mutter entband?»

«Das ist nicht leicht zu sagen. Wo sie sich jetzt aufhält, da gibt's
nichts zu tun für Hebammen; sie wird also wohl außer Dienst sein.»

«Was wollen Sie damit sagen?» fragte der Unbekannte finster.

«Daß sie im vergangenen Winter gestorben ist.»

Der Unbekannte sah ihn eine Zeitlang scharf an, sein Blick wurde
darauf zerstreut, und er schien in Gedanken versunken zu sein. Es war
zweifelhaft, ob ihm die erhaltene Kunde erfreulich oder unwillkommen
war, endlich aber schien er freier aufzuatmen, bemerkte, es käme wenig
darauf an, und stand auf, um sich zu entfernen.

Bumble besaß hinreichenden Scharfsinn, um sogleich zu gewahren, daß
sich ihm eine Gelegenheit eröffnet habe, Gewinn aus einem Geheimnisse
seiner besseren Hälfte zu ziehen. Er erinnerte sich des Todes der alten
Sally sehr wohl; war sie doch an dem Abend gestorben, an welchem er
Mrs. Corney seinen Antrag gemacht hatte; und obgleich ihm von Frau
Bumble noch immer nicht anvertraut worden war, was die Sterbende ihr
allein gebeichtet, so hatte er doch genug gehört, um zu wissen, daß
es sich auf etwas bezogen, das sich bei oder nach der Entbindung der
Mutter Oliver Twists ereignet hatte. Er sagte dem Unbekannten daher mit
geheimnisvoller Miene, daß die Alte, nach welcher er sich erkundigt,
kurz vor ihrem Tode eine andere Frau habe zu sich rufen lassen und
derselben Mitteilungen gemacht habe, die, wie er nicht ohne Grund
glaube, Licht in die Sache bringen könnten, um welche es sich handle.

«Wo kann ich die Frau sprechen?» fragte der Unbekannte, offenbar
überrascht, denn er ließ durchblicken, daß er lebhafte Befürchtungen
hegte, worin dieselben auch bestehen mochten.

«Nur durch meine Vermittlung», erwiderte Bumble.

«Wann?» fragte der Unbekannte in großer Aufregung weiter.

«Morgen.»

«Abends um neun Uhr», sagte der Unbekannte und schrieb mit etwas
zitternder Hand die Adresse eines abgelegenen Hauses auf. «Bringen
Sie sie abends um neun Uhr zu mir. Ich brauche Ihnen nicht zu sagen,
insgeheim, denn es ist Ihr Vorteil.»

Er ging darauf mit Bumble zur Tür, bezahlte den Wirt, bemerkte, daß sie
sich hier trennen müßten, schärfte dem Armenhausverwalter noch einmal
Pünktlichkeit ein und ging. Bumble sah auf die Adresse; sie hatte
keinen Namen. Er folgte daher dem Unbekannten nach, um ihn darum zu
befragen und berührte seinen Arm.

«Was soll das?» fuhr ihn der Unbekannte, sich rasch umdrehend, an.
«Warum folgen Sie mir nach?»

«Ich muß doch wissen, nach wem ich zu fragen habe», sagte Bumble; «darf
ich nicht um Ihren Namen bitten?»

«Monks!» erwiderte der Unbekannte und entfernte sich mit eiligen
Schritten.




38. Kapitel.

    Was sich zwischen Mr. und Mrs. Bumble und Monks bei ihrer
    nächtlichen Zusammenkunft begab.


Es war ein schwüler Sommerabend; die Wolken, welche den ganzen Tag
gedroht hatten, dehnten sich zu einer breiteren und dichteren Masse
aus, aus welcher schon dicke Regentropfen herabfielen, und schienen
ein heftiges Gewitter zu verkünden, als sich Mr. und Mrs. Bumble aus
einer der Hauptstraßen der Stadt nach einer kleinen Kolonie zerstreut
stehender und verfallener Häuser wandten, die etwa anderthalb Meilen
entfernt sein mochten, und in einer sumpfigen Niederung am Themseufer
erbaut waren. Sie hatten sich beide in schäbige Mäntel eingehüllt,
vielleicht sowohl um sich vor dem Regen zu schützen, wie um unbemerkt
zu bleiben. Mr. Bumble trug eine Laterne, in welcher jedoch kein Licht
brannte, und ging ein paar Schritte voran, als hätte er -- denn der Weg
war schmutzig -- seiner Frau den Vorteil verschaffen wollen, in seine
breiten Fußstapfen zu treten. Sie schritten in tiefem Stillschweigen
dahin, Mr. Bumble sah sich bisweilen um, als wenn er sich hätte
überzeugen wollen, ob Mrs. Bumble auch nachfolgte, worauf er ebensooft,
sie hinter sich gewahrend, seine Schritte wieder beschleunigte.

Ihr Bestimmungsort konnte keineswegs ein zweideutiger heißen, denn
er war längst als die Wohnung von verrufenem und verwegenem Gesindel
bekannt, das hauptsächlich von Diebstählen und Räubereien lebte. Es war
ein Haufen elender Baracken, in deren Mitte am Uferrande ein großes
Gebäude stand, das ehemals zu Fabrikzwecken der einen oder anderen Art
gedient und den Hüttenbewohnern umher wahrscheinlich Beschäftigung
gegeben hatte. Es war indes seit langer Zeit verfallen, und die Ratten,
die Würmer und die Feuchtigkeit hatten das Pfahlwerk morsch gemacht,
auf welchem es ruhte, so daß schon ein beträchtlicher Teil des Ganzen
unter das Wasser gesunken war, während der wankende und über den
finsteren Strom hinüberlehnende Rest nur auf eine günstige Gelegenheit
zu warten schien, dasselbe Schicksal zu teilen.

Vor diesem Gebäude stand das würdige Paar still, als eben das erste
Rollen des entfernten Donners vernehmbar wurde, und der Regen mit
Heftigkeit niederzustürzen anfing.

«Es muß hier irgendwo sein», sagte Bumble, auf einen Papierstreifen
blickend, den er in der Hand hielt.

«Wer da?» ertönte eine Stimme von oben.

Bumble blickte empor und sah jemanden aus dem zweiten Stockwerke
herunterschauen.

«Eine Minute Geduld,» rief die Stimme, «ich werde sogleich bei Ihnen
sein.»

«Ist das der Mann?» fragte Frau Bumble, und ihr Eheherr nickte bejahend.

«Vergiß nicht, was ich dir gesagt habe,» fuhr die Dame fort, «und
sprich so wenig wie nur irgend möglich, denn du wirst uns sonst gleich
verraten.»

Mr. Bumble, der an dem Hause mit sehr bänglichen Blicken emporgeschaut
hatte, stand im Begriff, einige Zweifel auszusprechen, ob es überhaupt
rätlich sei, sich noch zu dieser Stunde auf das Abenteuer einzulassen,
als er durch Monks daran gehindert wurde, der eine kleine Tür öffnete,
vor welcher sie standen, und ihnen winkte hereinzutreten. «Geschwind!»
rief er ungeduldig, und mit dem Fuße stampfend. «Haltet mich hier nicht
auf!»

Frau Bumble, welche anfangs gezögert hatte, ging keck hinein, und ihr
Eheherr, der sich schämte oder fürchtete, zurückzubleiben, folgte ihr
nach, jedoch offenbar mit großer Unruhe und ohne jene Würde, die ihn
sonst stets vornehmlich zu charakterisieren pflegte.

«Was zum Teufel stehen Sie da draußen und ließen sich naß regnen?»
sagte Monks zu ihm, nachdem er die Tür wieder verriegelt hatte.

«Wir -- wir kühlten uns ein wenig ab», stotterte Bumble, furchtsam
umherblickend.

«Kühlten sich ein wenig ab!» entgegnete Monks. «Aller Regen, der jemals
vom Himmel herabfiel oder noch herabfallen soll, wird nicht so viel
höllisches Feuer auslöschen, wie ein Mann mit sich umhertragen kann.
Glauben Sie nicht, daß Sie sich so leicht abkühlen können.»

Mit diesen angenehmen Worten und mit einem finsteren, stieren Blick
wandte sich Monks zu Frau Bumble, die, obwohl sonst nicht so leicht
einzuschüchtern, dennoch die Augen vor ihm auf den Boden heften mußte.
«Ist dies die Frau?» fragte Monks.

«Hm! Ja!» antwortete Mr. Bumble, eingedenk der Warnung seiner Gattin.

«Sie glauben vielleicht, daß Frauen keine Geheimnisse verschweigen
können!» nahm Frau Bumble das Wort und blickte dabei Monks wieder
dreist und forschend an.

«Ich weiß, daß sie allezeit eins verschweigen, bis es an den Tag
gekommen ist», erwiderte Monks verächtlich.

«Und was ist das für ein Geheimnis?» fragte Frau Bumble in demselben
zuversichtlichen Tone.

«Der Verlust ihres guten Namens», sagte Monks; «und ebenso fürchte ich
nicht, daß eine Frau ihr Geheimnis ausschwatzt, wenn das Ausschwatzen
dahin führen kann, daß sie gehängt oder deportiert wird. Verstanden?»

«Nein», versetzte die Dame, sich ein wenig verfärbend.

«Freilich,» sagte Monks spöttisch, «wie könnten Sie mich auch
verstehen!» Er blickte die Eheleute halb höhnisch und halb grollend an,
winkte ihnen abermals, ihm nachzufolgen, eilte durch das große, jedoch
niedrige Zimmer voran und wollte eben eine steile Treppe oder vielmehr
Leiter hinaufsteigen, als der helle Glanz eines Blitzes durch die
Öffnung herabfuhr und ein Donnerschlag erfolgte, der das gebrechliche
Gebäude in seinem Grunde erschütterte.

«Hören Sie!» rief er, zurückschreckend aus. «Hören Sie, wie es prasselt
und rollt, als ob es durch tausend Höhlen widerhallte, wo sich die
Teufel davor verstecken. Fluch über den Lärm! Ich hasse ihn.»

Er schwieg einige Augenblicke, entfernte plötzlich die Hände von seinem
Gesicht, und Mr. Bumble gewahrte zu seinem unaussprechlichen Schrecken,
daß es fast kreideweiß und ganz verzerrt war.

«Ich leide bisweilen an diesen Zufällen,» sagte Monks, die Bestürzung
des Armenhausverwalters bemerkend, «und dann und wann werden sie
durch den Donner hervorgerufen. Achten Sie nicht darauf; es ist für
diesmal vorüber.» Mit diesen Worten ging er voran, erklomm die Treppe,
verschloß hastig die Fensterläden des Gemaches, in welches er das
Ehepaar führte, und ließ eine an einer Leine und einer Rolle an einem
der Deckenbalken hängende Laterne herunter, die ein mattes Licht auf
einen alten Tisch und drei an denselben gestellte Stühle warf. Als sie
sich gesetzt hatten, sagte er: «Je eher wir zur Sache kommen, desto
besser ist's für uns alle. Weiß die Frau, worauf sich unser Geschäft
bezieht?»

Die Frage war an Mr. Bumble gerichtet, allein Mrs. Bumble nahm sogleich
das Wort und erklärte, daß sie mit dem Zwecke der Zusammenkunft
vollkommen bekannt sei.

«Er sagte, Sie wären bei der alten Hexe an dem Abend gewesen, da sie
starb, und sie hätte Ihnen etwas anvertraut --»

«Was die Mutter des Knaben betraf, den Sie nannten», unterbrach ihn
Frau Bumble. «Ja, Sir.»

«Die erste Frage», sagte Monks, «ist die, worin bestand ihre
Mitteilung?»

«Das ist die zweite Frage», bemerkte Frau Bumble mit großer Ruhe. «Die
erste ist die, was wohl der Preis des Geheimnisses sein mag?»

«Wer zum Teufel kann das sagen, ohne zu wissen, worin es besteht?»
lautete Monks' Gegenfrage.

«Ich bin überzeugt, niemand besser als Sie», antwortete Frau Bumble,
der es, wie ihr Gatte aus hinreichender Erfahrung bezeugen konnte,
keineswegs an Herzhaftigkeit gebrach.

«Hm!» sagte Monks bedeutsam und mit einem begierigen und lauernden
Blick; «handelt es sich denn um etwas Wertvolles?»

«Vielleicht -- o ja, vielleicht», antwortete Frau Bumble gelassen.

«Etwas, was man ihr abnahm», fuhr Monks eifrig fort; «etwas, was sie
trug -- etwas, was --»

«Sie tun am besten, wenn sie bieten», unterbrach ihn Frau Bumble. «Ich
habe schon gehört, um gewiß zu sein, daß Sie der Mann sind, für welchen
mein Geheimnis Wert hat.»

Mr. Bumble, den seine bessere Hälfte von dem Geheimnis noch nicht mehr
hatte wissen lassen, als er gleich zu Anfang gewußt, horchte diesem
Zwiegespräch mit vorgerecktem Halse und weit aufgerissenen Augen, die
er mit unverhohlenem Erstaunen bald auf seine Frau, bald auf Monks
heftete, und seine Spannung nahm womöglich noch zu, als der letztere
ernstlich nach der Summe fragte, welche für die Offenbarung des
Geheimnisses gefordert würde.

«Was ist es Ihnen wert?» fragte Frau Bumble ebenso kaltblütig wie
vorhin.

«Kann sein, daß es mir nichts oder daß es mir zwanzig Pfund wert ist»,
erwiderte Monks; «sprechen Sie und lassen Sie mich Ihre Forderung
wissen.»

«Legen Sie noch fünf Pfund zu; geben Sie mir fünfundzwanzig Pfund in
Gold,» versetzte Frau Bumble, «und ich sage Ihnen alles, was ich weiß
-- doch eher nicht.»

«Fünfundzwanzig Pfund!» rief Monks, sich zurückbeugend, aus.

«Ich sprach so deutlich, wie ich konnte,» entgegnete Frau Bumble, «und
die Summe ist auch nicht bedeutend.»

«Die Summe nicht bedeutend für ein erbärmliches Geheimnis, das
vielleicht der Rede nicht wert ist, wenn Sie es offenbart haben!» rief
Monks ungeduldig aus; «ein Geheimnis, das seit zwölf Jahren oder länger
vergessen oder begraben gelegen hat!»

«Solche Dinge halten sich gut und verdoppeln gleich gutem Weine häufig
ihren Wert durch die Zeit», bemerkte Frau Bumble mit der kalten
Entschlossenheit, die sie angenommen hatte; «und was das betrifft, daß
es begraben gewesen, so gibt es Leute, die, soviel wir wissen, noch
zwölftausend oder zwölf Millionen Jahre begraben liegen können und
endlich sonderbare Geschichten erzählen werden.»

«Wie aber, wenn ich für nichts zahle?» fragte Monks bedenklich zögernd.

«Sie können mir das Geld leicht wieder abnehmen», erwiderte die
Dame. «Ich bin ja nur eine Frau und allein und ohne Schutz in Ihrer
abgelegenen Wohnung.»

«Weder allein, meine Liebe, noch ohne Schutz», fiel Mr. Bumble mit
vor Angst bebender Stimme ein; «ich bin auch hier, meine Liebe. Und
außerdem,» fuhr er zähneklappernd fort, «und außerdem ist Mr. Monks zu
sehr Gentleman, um sich auch nur die mindeste Gewalttätigkeit gegen
Kirchspielpersonen zu erlauben. Mr. Monks weiß, daß ich nicht mehr in
der Blüte der Jahre und der Kraft stehe; allein er hat gehört -- hat
ohne Zweifel gehört, lieber Schatz, daß ich ein sehr entschlossener
Beamter und ungewöhnlich stark bin, wenn ich Veranlassung bekomme, mich
zusammenzunehmen. Ich brauche mich nur eben etwas zusammenzunehmen.»

Und als Mr. Bumble so sprach, machte er einen trübseligen Versuch, mit
trotziger Entschlossenheit nach seiner Laterne zu greifen, und zeigte
deutlich durch den in allen seinen Zügen sich malenden Schrecken, wie
es allerdings bei ihm nötig war, daß er sich ein wenig oder vielmehr
recht sehr zusammennehmen mußte, bevor er sich zu einer nur irgend
kriegerischen Demonstration herbeiließ, ausgenommen gegen Arme oder
andere wehrlose Personen.

«Du bist ein Narr,» entgegnete ihm seine Ehehälfte, «und kannst nichts
Besseres tun als den Mund halten.»

«Und ich werde ihm sogleich darauf schlagen, wenn er nicht leiser
spricht», sagte Monks zornig. «Er ist also Ihr Mann?»

«Er mein Mann!» kicherte Frau Bumble, der Frage ausweichend.

«Ich dachte es, als Sie beide hereinkamen», fuhr Monks fort, den
zornigen Blick gewahrend, den die Dame ihrem Eheherrn zuwarf. «Desto
besser; ich trage um so weniger Bedenken, mit Leuten zu unterhandeln,
wenn ich finde, daß sie von einem und demselben Willen beseelt sind.
Ich meine es ernstlich -- schauen Sie hier!»

Er zog einen Beutel aus der Tasche, zählte fünfundzwanzig Sovereigns
auf den Tisch und schob sie Frau Bumble hin.

«Nehmen Sie,» fuhr er fort, «und lassen Sie mich nun hören, was Sie zu
erzählen haben, sobald der verwünschte Donnerschlag vorüber ist, der,
ich fühl's, gerade über dem Hause loswettern wird.»

Sobald das Donnergeroll vorüber war, hob Monks das Gesicht vom Tische
empor und beugte sich zu Frau Bumble hinüber, um begierig zu hören, was
sie sagen würde. Auch das Ehepaar lehnte sich über den kleinen Tisch,
so daß die Köpfe von allen dreien sich berührten. Das auf sie gerade
herunterfallende matte Licht der hängenden Laterne ließ ihre Gesichter
noch bleicher und gespenstischer erscheinen, und sie sahen um so
unheimlicher aus, als rings umher die tiefste Finsternis sie umgab.

«Als die Frau, die wir die alte Sally nannten, starb,» hub Frau Bumble
flüsternd an, «war ich mit ihr allein.»

«War niemand dabei?» fragte Monks mit demselben hohlen Geflüster;
«keine Kranke oder Verrückte in einem anderen Bette? -- keine Seele,
welche hören, vielleicht verstehen konnte?»

«Wir waren ganz allein», versicherte Frau Bumble; «ich und sonst
niemand stand an ihrem Bette, als sie im Sterben lag. Sie sprach von
einer jungen Frauensperson, die einige Jahre zuvor einem Kinde das
Leben gegeben hätte, und zwar nicht bloß in demselben Zimmer, sondern
auch in demselben Bette, in welchem die Sterbende lag.»

«Fürwahr!» sagte Monks mit bebender Lippe und über seine Schulter
blickend. «Teufel! Wie doch die Dinge zuletzt kommen können!»

«Das Kind war dasselbe, das Sie ihm gestern abend nannten», fuhr Frau
Bumble, nachlässig nach ihrem Manne hindeutend, fort; «und die alte
Sally hat seine Mutter bestohlen.»

«Bei ihren Lebzeiten?» fragte Monks.

«Nein, als sie gestorben war», erwiderte Frau Bumble mit einigem
Schaudern. «Sie bestahl die Leiche, nachdem dieselbe eine solche
geworden war, und was sie nahm, war eben das, was die sterbende Mutter
in ihren letzten Atemzügen sie gebeten hatte, um des Kindes willen
aufzubewahren.»

«Verkaufte sie es?» fiel Monks in der größten Spannung ein; «hat sie es
verkauft? -- Wo? -- Wann? -- An wen? -- Vor wie langer Zeit?»

«Als sie mir mit großer Mühe gesagt hatte, was sie getan, sank sie
zurück und starb.»

«Und sagte weiter nichts mehr?» rief Monks mit einer Stimme, die nur
um so wütender ertönte, je gewaltsamer er sie zu dämpfen suchte. «Es
ist eine Lüge! Ich werde mich nicht hinter das Licht führen lassen. Sie
sagte mehr -- ich morde Sie beide, wenn ich nicht erfahre, was es war!»

«Sie sagte kein Sterbenswörtchen mehr,» entgegnete Frau Bumble, allem
Anscheine nach durch Monks' Heftigkeit nicht im mindesten erschreckt,
was ihr Mann augenscheinlich in einem desto höheren Grade war; «sie
faßte aber krampfhaft mit der einen Hand mein Kleid, und ich fand,
als sie tot war, und als ich ihre Hand mit Gewalt losmachte, einen
schmutzigen Papierstreifen darin.»

«Was enthielt er?» unterbrach Monks, sich vorbeugend.

«Nichts; es war ein Schein von einem Pfandleiher.»

«Worüber?»

«Das werde ich Ihnen seinerzeit schon sagen. Ich muß glauben, sie hatte
das Geschmeide, über dessen Empfang der Papierstreifen ausgestellt
war, einige Zeit aufbewahrt, um größeren Gewinn daraus zu ziehen, es
sodann verpfändet und dem Pfandleiher jedes Jahr die Zinsen bezahlt, um
es wieder einlösen zu können, wenn es etwa zu einer Entdeckung führen
sollte. Dies war jedoch nicht geschehen, und sie starb mit dem Scheine
in der Hand, der nach einigen Tagen verfallen sein würde, und ich löste
das Pfand ein, weil ich glaubte, dereinst noch einmal Nutzen daraus
ziehen zu können.»

«Wo haben Sie es?» fragte Monks hastig.

«Hier ist es», erwiderte Frau Bumble und warf eilig, als wenn sie
froh wäre, sich davon zu befreien, ein kleines ledernes Beutelchen
auf den Tisch; Monks bemächtigte sich desselben begierig und öffnete
es mit zitternden Händen. Es enthielt ein kleines goldenes Medaillon,
in welchem sich zwei Haarlocken und ein einfacher goldener Trauring
befanden.

«Auf der Innenseite ist der Name Agnes zu lesen», sagte Frau Bumble.
«Für den Zunamen ist ein Raum offen gelassen, und dann folgt das Datum
von einem Tage in dem Jahre vor der Geburt des Kindes, das ich in
Erfahrung gebracht habe.»

«Und das ist alles?» fragte Monks nach einer genauen und eifrigen
Untersuchung des kleinen Beutels.

«Ja,» antwortete Frau Bumble, und ihr Eheherr atmete lang und tief,
als wenn er sich freute, daß alles vorüber wäre, ohne daß Monks die
fünfundzwanzig Pfund zurückforderte. Er faßte jetzt so viel Mut, um
endlich den Schweiß abzuwischen, der ihm vom Anfange der Unterredung an
über die Stirn und Wangen hinabgeträufelt war.

«Ich weiß nichts von der Geschichte außer dem, was ich mutmaßen kann,»
nahm seine Frau nach einem kurzen Stillschweigen wieder das Wort, «und
begehre auch nichts zu wissen, denn es ist sicherer. Darf ich Ihnen
aber ein paar Fragen vorlegen?»

«Das können Sie», sagte Monks mit einiger Verwunderung; «ob ich aber
antworte oder nicht, ist eine andere Frage.»

«Was ihrer drei macht», bemerkte Mr. Bumble, ein wenig
Scherzhaft-Witziges einschaltend.

«War es das, was Sie von mir zu bekommen erwarteten?» fragte die Dame.

«Ja», erwiderte Monks. «Die zweite Frage?» --

«Was denken Sie damit zu tun -- kann es gegen mich gebraucht werden?»

«Niemals,» sagte Monks, «und auch nicht gegen mich. Sehen Sie hier;
aber bewegen Sie sich keinen Schritt vorwärts, oder Ihr Leben ist
keinen Strohhalm wert!» Er schob bei diesen Worten plötzlich den Tisch
zur Seite und öffnete eine große Falltür dicht vor den Füßen Mr.
Bumbles, der sich in größter Hast mehrere Schritte zurückzog. «Schauen
Sie hinunter», sagte Monks, die Laterne in die Öffnung hinablassend;
«fürchten Sie nichts. Ich hätte Sie ganz unbemerkt hinunter spedieren
können, als Sie darüber saßen, wenn es meine Absicht gewesen wäre.»

Frau Bumble trat ermutigt an die Öffnung, und sogar ihr Eheherr
wagte es, von Neugierde getrieben, dasselbe zu tun. Das vom Regen
angeschwollene trübe Wasser rauschte unten so gewaltig, daß sich alle
anderen Töne in seinem Geräusche verloren. Es war an der Stelle vormals
eine Wassermühle gewesen, und das Pfahlwerk und die sonstigen Überreste
derselben hielten das Wasser nur auf, um seinen Andrang und das Brausen
noch zu verstärken.

«Wenn man hier eine Leiche hinunterwürfe, wo würde sie morgen früh
sein?» fragte Monks, die Laterne in dem finsteren Schlunde hin und her
schwingend.

«Zwölf Meilen weit unten im Strome und obendrein in Stücke gerissen»,
erwiderte Bumble, bei dem bloßen Gedanken zurückbebend.

Monks nahm den kleinen Beutel, band ihn fest an ein daliegendes
bleiernes Gewicht und warf ihn in das Wasser hinunter; man hörte,
wie er hineinfiel, alle drei sahen einander an und schienen freier
aufzuatmen. Monks verschloß die Falltür wieder.

«So!» sagte er. «Wenn die See ihre Toten jemals zurückgibt, wie Bücher
sagen, daß sie es werde -- so wird sie doch ihr Gold und Silber samt
jenem Plunder für sich behalten. Wir haben einander nichts mehr zu
sagen und können unserem angenehmen Zusammensein ein Ende machen.»

«Allerdings, allerdings», bemerkte Mr. Bumble mit großem Eifer.

«Sie werden doch aber reinen Mund halten?» fragte Monks mit einem
drohenden Blick. «Für Ihre Frau bin ich nicht besorgt.»

«Sie können sich auf mich verlassen, junger Mann», antwortete Bumble,
sich unter fortwährenden, unendlich höflichen Verbeugungen der Leiter
nähernd. «Um jedermanns willen, und Sie wissen, auch um meinetwillen,
Mr. Monks.»

«Ich freue mich um Ihretwillen, Sie so sprechen zu hören», entgegnete
Monks. «Zünden Sie Ihre Laterne an, und machen Sie sich davon, so
schnell Sie können.»

Diese Aufforderung kam sehr zur rechten Zeit, denn Mr. Bumble würde,
wenn er sich noch einmal verbeugt und dann noch einen einzigen Schritt
zurückgetan hätte, unfehlbar hinuntergestürzt sein. Er zündete seine
Laterne an, stieg schweigend hinab, und seine Frau folgte ihm. Monks
folgte zuletzt, nachdem er einige Augenblicke gehorcht hatte, ob sich
auch keine anderen Laute vernehmen ließen, als die des Wasser- und
Regengeräusches. Sie gingen langsam und vorsichtig durch das Zimmer
im Erdgeschoß, denn Monks erschrak über jeden Schatten, und Bumble
hielt seine Laterne einen Fuß über dem Boden und blickte fortwährend
angstvoll nach versteckten Falltüren umher. Monks öffnete ihnen leise
die Tür, und das Ehepaar trat in die Finsternis hinaus, nachdem es von
seinem geheimnisvollen Bekannten durch ein Kopfnicken Abschied genommen
hatte.

Sobald der Armenhausverwalter und seine Gattin fort waren, rief Monks,
der einen unüberwindlichen Widerwillen gegen das Alleinsein zu hegen
schien, einen Knaben, der irgendwo versteckt gewesen sein mußte, befahl
ihm, mit der Laterne voranzugehen, und kehrte in das Gemach zurück, das
er soeben verlassen hatte.




39. Kapitel.

    In welchem alte Bekannte auftreten und Fagin und Monks die Köpfe
    zusammenstecken.


An dem Abende, der auf die im vorigen Kapitel erzählte Unterredung der
drei wackeren Leute folgte, erwachte Sikes aus seinem Schlummer und
fragte schlaftrunken, welche Zeit es wäre. Das Zimmer, in welchem er
sich befand, war keins von denen, die er vor der Chertseyer Expedition
bewohnt hatte, obgleich es sich in einem Hause nicht weit von seiner
früheren Wohnung befand. Es war allem Anschein ein weit schlechteres
Gemach und erhielt nur durch ein einziges Dachfenster Licht, das auf
eine enge und schmutzige Gasse hinausging. Auch fehlte es nicht an
mannigfachen anderen Anzeichen, daß Mr. Sikes zur äußersten Dürftigkeit
herabgesunken war, was auch durch sein bleiches und abgemagertes
Aussehen bestätigt wurde.

Der Einbrecher lag auf seinem Bett, in einen großen weißen Mantel
gehüllt und mit einem Gesicht, das durch seine leichenhafte Blässe und
einen mindestens eine Woche alten, stachligen, schwarzen Bart nichts
weniger als verschönt war. Sein Hund saß neben dem Bett, bald seinen
Herrn mit ernsten Augen anblickend, bald die Ohren spitzend und ein
dumpfes Knurren ausstoßend, sobald ein Geräusch auf der Straße oder in
dem unteren Teile des Hauses seine Aufmerksamkeit erregte.

An dem Fenster mit Ausbesserung eines dem Einbrecher gehörenden alten
Kleidungsstückes beschäftigt, saß Nancy, welche gleichfalls so blaß und
erschöpft von Hunger und Wachen aussah, daß man sie kaum anders als
an der Stimme erkannt haben würde, als sie Sikes' Frage beantwortete.
«Noch nicht lange sieben vorüber», sagte sie. «Wie befindet Ihr Euch
heute abend, Bill?»

«So schwach wie Wasser», erwiderte er mit einem seiner gewöhnlichen
Flüche. «Komm her, reich' mir die Hand und hilf mir von dem verdammten
Bette.»

Sikes' Laune war durch seine Krankheit nicht freundlicher geworden,
denn während ihn Nancy emporhob und nach einem Stuhle leitete, murmelte
er Flüche über ihr Ungeschick und schlug sie.

«Plärrst du?» sagte er. «Laß das Winseln bleiben! Wenn du nichts
Besseres weißt, so troll' dich lieber. Hörst du?»

«Freilich hör' ich», antwortete das Mädchen, das Gesicht abwendend und
sich zu einem Lachen zwingend. «Was fällt Euch denn jetzt wieder ein,
Bill?»

«Hast dich 'nes Bessern besonnen?» sagte Sikes finster, die in ihrem
Auge zitternde Träne gewahrend. «Um so besser für dich.»

«Oh, Ihr könnt heut' abend nicht schlimm gegen mich sein, Bill»,
versetzte sie, die Hand auf seine Schulter legend.

«Warum nicht?» fuhr er sie an.

«Wie viele, viele Nächte,» sagte sie mit einer Regung von
Frauenzärtlichkeit, die sogar dem Ton ihrer Stimme eine gewisse
Weichheit gab, -- «wie viele, viele Nächte hab' ich geduldig bei Euch
gesessen und Euch gepflegt und gewartet, als ob Ihr ein Kind gewesen
wäret; und Ihr würdet mich sicher nicht behandelt haben, wie Ihr's eben
tatet, wenn Ihr daran gedacht hättet; nicht wahr, Bill? Sprecht nur ein
Wort -- sagt nein.»

«Nun ja, ich hätt's nicht getan», sagte Sikes. «Aber Gott verdamm'
mich, die Dirne winselt schon wieder!»

«'s ist nichts», seufzte Nancy, sich auf einen Stuhl werfend. «Kümmert
Euch nur nicht um mich, und es wird bald vorüber sein.»

«Was wird vorüber sein?» fragte Sikes zornig. «Was hast du jetzt wieder
für Dummheiten vor? Steh' auf, mach' dir zu schaffen und bleib mit
deinen Weiberpossen zu Haus!»

Zu jeder anderen Zeit würde diese Aufforderung und der Ton, in welchem
sie ausgesprochen wurde, die beabsichtigte Wirkung gehabt haben; allein
Nancy war in der Tat kraftlos und erschöpft, ließ den Kopf auf die
Stuhllehne sinken und wurde ohnmächtig, ehe noch Sikes die angemessenen
Flüche ausstoßen konnte, mit welchen er unter ähnlichen Umständen seine
Drohungen zu würzen pflegte. Er wußte nicht recht, was er tun sollte,
denn Nancys Ohnmachten pflegten von der heftigsten Art zu sein; er nahm
daher seine Zuflucht zu ein wenig Gotteslästerung und rief nach Hilfe,
als sich das Mittel vollkommen unwirksam zeigte.

«Was gibt es, mein Lieber?» fragte der Jude hereinblickend.

«Kannst der Dirne nicht beispringen?» rief ihm Sikes ungeduldig zu.
«Steh' nicht da und schwatz', gaff' mich nicht an!»

Fagin eilte mit einem Ausrufe der Verwunderung, Nancy Beistand zu
leisten, während Mr. John Dawkins (sonst genannt der gepfefferte
Baldowerer), der seinem ehrwürdigen Freunde in das Zimmer gefolgt
war, hastig ein Bündel niederlegte, Master Charley Bates, der dicht
hinter ihm war, eine Flasche aus der Hand riß, sie im Nu mit den Zähnen
entkorkte und der Patientin einige Tropfen daraus eingoß, jedoch erst,
nachdem er selbst gekostet, um einen etwaigen Irrtum zu verhüten.

«Blase ihr 'n Bissel frische Luft mit dem Blasebalge zu, Charley,»
sagte er, «und Ihr, Fagin, klapst ihr die Hände, während Bill ihr die
Kleider lockert.»

Da alle sehr eifrig waren, besonders Master Bates, dem seine Rolle
der köstlichste Spaß zu sein schien, so kam Nancy nach kurzer Zeit
allmählich wieder zu sich selbst, wankte nach einem Stuhle am Bett,
verbarg ihr Gesicht in den Kissen und überließ es Sikes, ohne alle
Einmischung von ihrer Seite, den Neuangekommenen seine Meinung über sie
und ihr unerwartetes Erscheinen auszudrücken.

«Welcher böse Wind hat Euch denn hierher geblasen?» fragte er Fagin.

«Gar kein böser Wind, mein Lieber,» antwortete der Jude; «denn ein
böser Wind bläst zu niemandem Gutes, und ich habe mitgebracht etwas
Gutes, das Ihr Euch werdet freuen zu schaun. Baldowerer, mein Lieber,
öffne das Bündel und gib Bill, wofür wir haben ausgegeben all unser
Geld.»

Der Gepfefferte band das Bündel auf, und Charley Bates leerte es unter
Lobsprüchen des Inhalts.

«Schaut nur, Bill,» sagte der junge Herr, «solch 'ne Kaninchenpastete,
von so zarten Tierchen, daß einem sogar die Knochen auf der Zunge
zerschmelzen; -- und hier den prächt'gen Tee -- und den Zucker -- und
das Brot -- und die frische Butter -- den Gloucesterkäs -- und vor
allen Dingen, was sagt Ihr hierzu?»

Er stellte bei diesen Worten eine wohlverkorkte Weinflasche auf
den Tisch, während Dawkins aus der Flasche, die er vorhin Charley
entrissen, dem Patienten ein Glas Branntwein einschenkte, das von
demselben sogleich auf einen Zug geleert wurde.

«Das wird Euch bekommen, wird Euch bekommen, Bill», sagte der Jude,
sich vergnügt die Hände reibend.

«Bekommen?» rief Sikes aus. «Ich hätte zwanzigmal umkommen können, eh'
du 'nen Finger für mich gerührt hättest. Was soll das bedeuten, du
falscher Schuft, daß du einen in 'nem solchen Zustande länger als drei
Wochen im Stich lässest?»

«Hört, Kinder, hört ihn nur!» sagte der Jude achselzuckend; «hört, was
er sagt, da wir kommen eben und ihm bringen alle die prächtigen Sachen.»

«Die Sachen sind in ihrer Art ganz gut», bemerkte Sikes, durch einen
Blick nach dem wohlbesetzten Tische ein wenig besänftigt; «aber womit
kannst du dich entschuldigen, daß du mich hier krank, ohne Geld und
entblößt von allem hast liegen lassen und dich die ganze Zeit nicht
mehr um mich bekümmert hast, als wenn ich nicht besser wär' wie der
Hund da?»

«Ich bin gewesen aus London, mein Lieber, länger als eine Woche»,
erwiderte der Jude.

«Und wo warst du die anderen vierzehn Tage,» fragte Sikes, «wo du mich
hast hier liegen lassen wie 'ne Ratt' in ihrem Loche?»

«Konnt's nicht ändern, Bill», antwortete Fagin; «kann mich nicht
einlassen auf die Gründe vor so vielen Ohren; aber, auf meine Ehre, ich
konnt's nicht ändern.»

«Worauf?» schnaubte ihn Sikes mit der äußersten Verachtung an. «Jungen,
schneid' mir einer von euch ein Stück Pastete ab, daß ich den Geschmack
von seiner Ehr' aus dem Munde los werde, oder ich ekle mich daran zu
Tode.»

«Seid nur nicht unwirsch, mein Lieber», erwiderte der Jude sehr
unterwürfig. «Ich hab' Euch vergessen nicht, Bill; niemals, Bill.»

«Oh, ich will selbst darauf schwören», fiel Sikes mit dem bittersten
Lächeln ein. «Du gehst deinen Geschäften nach, während ich hier im
Fieber liege. Ich hab' bald dies, bald das für dich tun müssen, solang
ich gesund und auf'n Beinen war, und hab's spottwohlfeil getan und bin
arm dabei geblieben und hätte sterben und verderben müssen, wär' die
Dirn' nicht gewesen.»

«Ganz recht, Bill,» sagte der Jude, Sikes' letzte Äußerung begierig
auffassend, «wär' nicht gewesen die Dirne! Wer aber hat sie erzogen als
der arme, alte Fagin, und hättet Ihr sie gehabt ohne mich?»

«Er hat ganz recht», rief Nancy aus, hastig näherkommend. «Laßt ihn
zufrieden.»

Nancys Erscheinen gab dem Gespräch eine andere Wendung, denn die Jungen
begannen auf einen Wink des schlauen alten Juden hin ihr Branntwein
einzuschenken, während Fagin mit Aufbietung all seines Witzes Sikes
endlich in eine bessere Laune brachte, indem er sich stellte, als
betrachtete er seine Drohungen als kleine, harmlose Scherze und
außerdem von Herzen über ein paar rohe Späße lachte, zu denen sich der
andere, nachdem er wiederholt der Branntweinflasche zugesprochen hatte,
herabließ.

«Das ist alles ganz gut,» sagte Sikes endlich, «aber ich muß heute
abend noch Geld von dir haben.»

«Ich habe nichts, habe gar nichts bei mir, Bill», wandte der Jude ein.

«Dann hast du desto mehr zu Hause,» sagte Sikes, «und ich muß darum was
haben.»

«Desto mehr!» rief Fagin, die Hände emporhebend, aus. «Ich habe nicht
soviel, um nur --»

«Ich weiß nicht, wieviel du hast,» unterbrach ihn Sikes, «und du magst
es selbst wohl nicht wissen, denn es wird 'ne gute Zeit dazu gehören,
es zu zählen; aber gleichviel, ich muß und muß noch heut' abend Geld
haben.»

«Nun gut, schon gut», entgegnete Fagin seufzend; «so will ich den
Baldowerer schicken.»

«Das sollst du bleiben lassen», sagte Sikes. «Der Gepfefferte ist ein
gut Teil zu gepfeffert und würd' das Herkommen vergessen oder sich
vom Wege verlieren oder die Schuker[AP] baldowerten ihn, so daß er
verhindert wär', oder was er sonst für Ausflüchte ersänne. Nancy soll
mitgehen und 's holen, und ich will mich unterdes hinlegen und dormen.»

  [AP] Polizeidiener.

Nach vielem Markten und Feilschen kam endlich die Abrede zustande, daß
Sikes drei Pfund und vier Schillinge erhalten solle, worauf der Jude
mit seinen Zöglingen ging und Sikes sich niederlegte, um die Zeit bis
zu Nancys Rückkehr zu verschlafen. In der Wohnung des Juden saßen
Toby Crackit und Mr. Chitling beim fünfzehnten Spiele Cribbage, das
der letztere natürlich samt seinem fünfzehnten und letzten Sixpence
verlor. Mr. Crackit schien sich ein wenig zu schämen, mit einem jungen
Herrn sich eingelassen zu haben, der hinsichtlich seiner Stellung und
Geistesgaben so weit unter ihm war, gähnte, fragte nach Sikes und griff
zu seinem Hute.

«Niemand hier gewesen, Toby?» fragte der Jude.

«Kein lebendiges Bein», antwortete Mr. Crackit, an seinem Hemdkragen
zupfend. «Ihr müßtet eigentlich ein tüchtiges Stück Geld zahlen,
um mich dafür zu belohnen, daß ich Eu'r Haus solange gehütet. Gott
verdamm' mich, ich bin so dämlich wie ein Geschworener und wäre so
fest eingeschlafen wie in Newgate, wenn mich meine Gutmütigkeit nicht
bewogen hätte, mich mit dem jungen Menschen abzugeben. 's ist hier
schauderhaft langweilig gewesen.»

Er steckte bei diesen Worten seinen Gewinn mit einer Miene in die
Westentasche, als wenn es im Grunde tief unter seiner Würde wäre,
so kleine Münzen an sich zu nehmen, und entfernte sich mit seinem
gewöhnlichen renommistisch-gentilen Wesen. Tom Chitling sandte ihm
bewundernde Blicke nach und erklärte, daß er seinen Verlust um einer
solchen Bekanntschaft willen für nichts achte. Master Bates verspottete
ihn, worauf er Fagin zur Entscheidung aufforderte. Der Jude gab Dawkins
und Charley einen Wink und versicherte Tom, daß er ein sehr gescheiter
junger Mensch wäre.

«Und ist nicht Mr. Crackit eine grandige Sinze[AQ], Fagin?» fragte Tom.

  [AQ] Großer Herr, Gentleman.

«Freilich, freilich, mein Lieber.»

«Und ist's einem nicht 'ne Ehre, mit ihm Bekanntschaft zu haben?»

«Allerdings, mein Lieber. Die beiden sind nur eifersüchtig, weil er sie
nicht gönnt ihnen.»

«Seht ihr wohl?» rief Tom triumphierend. «Er hat mich ausgezogen, ich
kann aber hingehen und wieder was verdienen und noch mehr, sobald ich
nur will -- nicht wahr, Fagin?»

«Ja, ja, Tom,» erwiderte der Jude, «und je eher es geschieht, desto
besser. Also verloren mehr keine Zeit! Baldowerer, Charley, 's ist Zeit
für euch, auszugehen auf Massematten[AR] -- 's ist schon fast zehn und
noch nichts geschafft.»

  [AR] Geschäft, Unternehmen.

Der Baldowerer und Charley sagten Nancy gute Nacht und entfernten sich
unter mannigfachen Witzen auf Tom Chitlings Kosten, dessen Benehmen
jedoch ganz und gar nicht besonders auffällig oder ungewöhnlich gewesen
war; denn wie viele vortrefflich junge Gentlemen gibt es nicht, die
einen noch weit höheren Preis bezahlen als er, um in guter Gesellschaft
gesehen zu werden; und wie groß ist die Anzahl der die besagte gute
Gesellschaft bildenden feinen und vornehmen Herren, die ihren Ruf so
ziemlich auf dieselbe Weise begründen, wie der elegante Toby Crackit!

«Nun will ich dir holen das Geld, Nancy», sagte der Jude, als sie
fort waren. «Das ist nur der Schlüssel zu einem kleinen Schranke, wo
ich aufbewahre allerhand Schnurrpfeifereien, welche gebracht haben
die Jungens. Ich verschließe nie mein Geld, weil ich keins habe zu
verschließen. Das Geschäft geht schlecht, Nancy, und ich habe keinen
Dank davon, aber ich freue mich, das junge Volk zu sehen um mich --
pst!» unterbrach er sich, den Schlüssel hastig wegsteckend, «was war
das? -- horch!»

Nancy saß mit untergeschlagenen Armen am Tisch, und es schien ihr
vollkommen gleichgültig zu sein, ob jemand käme oder ginge und wer das
wäre, bis das Gemurmel einer Männerstimme ihr Ohr traf. Sobald sie die
Laute vernahm, legte sie mit Blitzesschnelle ihren Hut und Schal ab und
warf beides unter den Tisch. Gleich darauf drehte der Jude sich um, und
sie klagte mit matter Stimme, deren Ton gar sehr gegen ihre eben erst
bewiesene, von Fagin jedoch nicht bemerkte Hast und Heftigkeit abstach,
über Hitze.

«'s ist der Mann, den ich erwartete», sagte der Jude flüsternd und
offenbar verdrießlich über die Unterbrechung. «Er kommt jetzt herunter
die Treppe. Kein Wort von dem Gelde, Kind, in seiner Gegenwart. Er
bleibt nicht lange hier -- keine zehn Minuten, liebes Kind.» Er hielt
den knöchernen Zeigefinger auf die Lippen, ging mit dem Licht nach
der Tür und legte in dem Augenblick die Hand auf den Griff, als der
Besucher hastig eintrat. -- Es war Monks.

«Nur eine von meinen jungen Schülerinnen», sagte Fagin, als Monks, eine
Unbekannte erblickend, zurücktrat.

Nancy sah gleichgültig nach Monks hin und wandte die Blicke darauf von
ihm ab; als er die seinigen aber auf den Juden richtete, schaute sie
ihn abermals verstohlen, aber so scharf und forschend an, als wenn sie
plötzlich eine ganz andere geworden wäre.

«Neuigkeiten?» fragte der Jude.

«Große.»

«Und -- und -- gute?» fragte der Jude stockend weiter, als ob er
fürchtete, Monks dadurch zu reizen, daß er sich zu hoffnungsfroh zeigte.

«Zum wenigsten keine schlechten», erwiderte Monks lächelnd. «Ich bin
diesmal tätig genug gewesen. Laßt uns ein paar Worte allein reden.»

Nancy rückte näher an den Tisch heran, machte aber keine Miene, das
Zimmer zu verlassen, obwohl sie sah, daß Monks nach ihr hindeutete.
Der Jude, der vielleicht fürchtete, daß sie etwas von dem Gelde sagen
möchte, wenn er ihr befehle, hinauszugehen, wies stumm nach oben und
ging mit Monks hinaus.

«Nicht wieder in das höllische Loch, wo wir damals waren», hörte Nancy
den letzteren sagen, während beide die Treppe hinaufstiegen. Der Jude
lachte und erwiderte etwas, was sie nicht verstand. Dem Schalle der
Fußtritte nach schienen sie in das zweite Stockwerk hinaufzugehen.
Sie zog rasch die Schuhe aus, horchte in der größten Spannung an der
Tür und schlich, sobald sie keinen Laut mehr vernahm, vollkommen
geräuschlos nach. Es mochte eine Viertelstunde verflossen sein, als sie
ebenso leise in das Zimmer zurückkehrte, und gleich darauf kamen auch
die beiden Männer wieder die Treppe herunter. Monks entfernte sich aus
dem Hause, und als der Jude nach einiger Zeit mit dem Gelde hereintrat,
setzte das Mädchen eben den Hut auf, wie um sich zum Fortgehen
anzuschicken.

«In aller Welt, Nancy, wie blaß bist du!» rief Fagin erschreckend aus.
«Was hast du angefangen?»

«Nichts, das ich wüßte, ausgenommen, daß ich hier wer weiß wie lange
in dem engen Zimmer gesessen habe», antwortete sie im gleichgültigsten
Tone. «Gebt mir endlich das Geld und laßt mich fort.»

Fagin zählte es ihr seufzend in die Hand, sagte ihr gute Nacht, und
sie ging. Sobald sie sich auf der offenen Straße befand, setzte sie
sich auf die Stufen vor einer Haustür und schien, ganz betäubt und
erschöpft, außerstande zu sein, ihren Weg fortzusetzen. Plötzlich
sprang sie indes wieder auf, eilte nach einer ganz anderen Richtung
fort, als nach der, wo Sikes Wohnung lag, beschleunigte ihre Schritte
und lief endlich, so schnell ihre Füße sie tragen konnten. Sie mußte
nach einer Weile stillstehen, um Atem zu schöpfen, schien auf einmal
wieder zur Besinnung zu kommen und rang die Hände und brach in Tränen
aus, als ob sie sich bewußt geworden wäre, etwas nicht tun zu können,
was zu tun sie auf das sehnlichste wünschte.

Sei es, daß die Tränen ihr Erleichterung verschafften, oder daß sie
erkannte, wie gänzlich hoffnungslos ihre Lage war: genug, sie kehrte
wieder zurück und eilte fast ebenso schnell nach Sikes' Wohnung, sowohl
um die verlorene Zeit wieder einzubringen, als um gleichsam mit ihren
stürmisch-drängenden Gedanken Schritt zu halten.

Wenn sie noch Erregtheit verriet, als sie sich dem Diebe zeigte, so
gewahrte er dieselbe doch nicht, sondern schlummerte wieder ein,
nachdem er gefragt, ob sie das Geld mitgebracht habe, und eine
bejahende Antwort erhalten hatte.

Es war ein glücklicher Umstand für das Mädchen, daß Sikes Geld erhalten
hatte und daher am folgenden Tage durch Essen und Trinken fast
fortwährend beschäftigt wurde, was eine so wohltätige Wirkung auf seine
Stimmung äußerte, daß er weder Zeit noch Neigung hatte, sich um sie
und ihr Benehmen sonderlich zu bekümmern. Seinem luchsäugigen Freunde,
dem Juden, würde es nicht entgangen sein, daß sie mit der Ausführung
irgendeines verzweifelten Entschlusses umging; allein Sikes besaß
Fagins scharfe Beobachtungsgabe nicht, so daß Nancys ungewöhnliche
Erregtheit und Unruhe keinen Verdacht bei ihm erweckte.

Je näher der Abend kam, desto größer wurde ihre Unruhe, und als sie
in gespannter Erwartung neben ihm saß und darauf wartete, daß er sich
in den Schlaf tränke, wurden ihre Wangen so blaß, und es blitzte ein
so ungewöhnliches Feuer aus ihren Augen, daß Sikes endlich aufmerksam
darauf werden mußte. Er war matt vom Fieber, trank heißes Wasser zu
seinem Branntwein, um jenes minder entzündlich zu machen, und hatte
Nancy das Glas gereicht, um es zum dritten oder vierten Male von ihr
füllen zu lassen, als ihm ihre Blässe und das Feuer in ihren Augen
zuerst auffielen. Er starrte sie an, stützte sich auf den Ellbogen,
murmelte einen Fluch und sagte: «Du siehst ja wie eine Leiche aus, die
wieder zum Leben erwacht ist. Was hast du?»

«Was ich habe?» erwiderte sie. «Nichts. Warum seht Ihr mich so scharf
an?»

«Was ist das wieder für eine Albernheit?» fragte er, die Hand auf ihre
Schultern legend und sie unsanft schüttelnd. «Was ist das? Was soll das
bedeuten? Woran denkst du, Mädchen?»

«An vielerlei, Bill», erwiderte sie schaudernd und die Hände auf die
Augen drückend. «Aber was tut's?»

Der Ton der erzwungenen Heiterkeit, in welchem sie die letzteren Worte
gesprochen hatte, schien auf Sikes einen stärkeren Eindruck zu machen
als ihr wilder und starrer Blick vorher.

«Ich will dir was sagen», fuhr er verdrießlich fort. «Wenn du nicht
vom Fieber angesteckt bist und es jetzt selbst bekommst, so ist etwas
mehr als Gewöhnliches im Winde und obendrein was Gefährliches. Du
willst doch nicht hingehen und -- nein, Gott verdamm'! das kannst du
nimmermehr!»

«Was kann ich nimmermehr?» fragte das Mädchen.

«Es gibt,» murmelte Sikes, die Blicke auf sie heftend, «es gibt keine
zuverlässigere, treuere Dirne in der Welt als sie, oder ich würde ihr
vor drei Monaten die Kehle abgeschnitten haben. Sie kriegt das Fieber
-- das ist das ganze.»

Er leerte das Glas und forderte darauf seine Arznei. Nancy sprang rasch
auf, bereitete sie, den Rücken ihm zukehrend, und gab sie ihm ein.

«Jetzt setze dich hier an mein Bett», sagte er, «und nimm dein eigenes
Gesicht vor, oder ich ändere es so, daß du es selbst nicht wieder
erkennst, wenn du es brauchst.»

Sie tat nach seinem Geheiß, er faßte ihre Hand, sank auf das Kissen und
heftete die Augen auf ihr Gesicht. Sie fielen ihm zu, er öffnete sie
wieder, blickte starr umher und verfiel endlich in einen tiefen und
schweren Schlummer. Der Griff seiner Hand löste sich, der ausgestreckte
Arm fiel schlaff nieder, und Sikes lag da wie in dumpfer Betäubung.

«Der Schlaftrunk hat endlich gewirkt», murmelte sie; «doch vielleicht
ist es schon zu spät.»

Sie kleidete sich hastig an, blickte furchtsam umher, als wenn sie
trotz des Schlaftrunks jeden Augenblick erwartete, den Druck von Sikes'
schwerer Hand auf ihrer Schulter zu fühlen, beugte sich über das Bett,
küßte den Mund des Räubers, öffnete und verschloß geräuschlos die
Tür und eilte aus dem Hause. Ein Wächter rief halb zehn Uhr, und sie
fragte ihn, ob es schon lange nach halb zehn wäre. Er erwiderte, eine
Viertelstunde; sie murmelte: «und ich kann erst in einer Stunde dort
sein», und eilte rasch weiter.

Sie schlug die Richtung von Spitalfields nach Westend ein. Viele der
Läden in den engen Seitengassen, durch die sie ihr Weg führte, waren
schon geschlossen. Als es zehn schlug, wuchs ihre Unruhe, zumal da
sie vielfach durch das Gedränge in den belebteren Straßen aufgehalten
wurde. Sie eilte so ungestüm und rücksichtslos auf Gefahr jeder
Art weiter, daß sie von den Fußgängern für eine Verrückte gehalten
wurde. Als sie sich Westend näherte, nahm das Gedränge ab, und sie
beschleunigte ihre Schritte noch mehr. Endlich erreichte sie ihren
Bestimmungsort: ein schönes Haus in einer Straße nicht weit vom
Hydepark. Es schlug eben elf. Sie trat in den Hausflur. Der Sitz des
Türstehers war leer; sie blickte ungewiß umher und näherte sich der
Treppe.

«Zu wem wollen Sie, junges Frauenzimmer?» rief ihr ein wohlgekleidetes
Stubenmädchen, das eine Tür hinter ihr öffnete, nach.

«Zu einer Dame hier im Hause.»

«Einer Dame!» lautete die mit einem Blicke der Verachtung begleitete
Antwort. «Zu was für einer Dame?»

«Miß Maylie», sagte Nancy.

Das Mädchen, das jetzt Zeit gehabt hatte, die Fremde genauer anzusehen,
antwortete nur durch einen Blick tugendhafter Entrüstung und rief einen
Bedienten, dem Nancy ihre Bitte wiederholte. Er fragte nach ihrem Namen.

«Sie brauchen gar keinen zu nennen.»

«In was für 'ner Angelegenheit wollen Sie die Dame sprechen?»

«Ich muß sie sprechen -- das genügt.»

Der Bediente befahl ihr, sich aus dem Hause zu entfernen und schob sie
nach der Tür hin.

«Nehmen Sie sich in acht -- Sie werden mich nicht lebendig aus dem
Hause hinausschaffen!» rief sie. «Ist denn niemand hier, der einem
armen Mädchen den kleinen Dienst leistet, zu der Dame hinaufzugehen?»

Inzwischen hatte sich die Dienerschaft versammelt. Der gutmütige Koch
legte sich in das Mittel und forderte den Bedienten auf, das Mädchen
Miß Rose zu melden.

«Wozu denn aber?» antwortete der Bediente. «Sie werden doch nicht
glauben, daß die junge Dame eine solche Person vorlassen wird?»

Diese Anspielung auf Nancys verdächtigen Stand erregte ein gewaltiges
Maß tugendsamer Entrüstung bei vier Dienstmädchen, welche mit
großer Lebhaftigkeit erklärten, das Geschöpf sei eine Schande ihres
Geschlechts, und darauf bestanden, sie ohne Gnade auf die Straße zu
werfen.

«Machen Sie mit mir, was Ihnen beliebt,» sagte das Mädchen, zu den
Bedienten sich wendend, «nur tun Sie erst, was ich verlange; und ich
fordere Sie auf, meine Botschaft um Gottes willen auszurichten.»

Der weichherzige Koch trat jetzt vermittelnd dazwischen, und das Ende
war, daß der Mann, der zuerst zum Vorschein gekommen, die Meldung
übernahm.

«Was soll ich meiner Herrschaft sagen?» fragte er.

«Daß ein junges Mädchen Miß Maylie unter vier Augen zu sprechen
wünscht», erwiderte Nancy; «und -- daß die junge Dame, wenn sie nur
das erste Wort anhören will, sogleich erkennen wird, ob sie das, was
ich anzubringen habe, noch ferner anhören muß, oder mich als eine
Betrügerin vor die Tür werfen lassen soll.»

«Meiner Six!» erwiderte der Bediente. «Sie sind Ihrer Sache ja sehr
gewiß.»

«Bringen Sie nur mein Anliegen vor, und lassen Sie mich den Bescheid
wissen», entgegnete das Mädchen fest.

Der Bediente eilte hinauf, und Nancy stand bleich, fast atemlos und mit
zuckenden Lippen da, als die sehr hörbaren Ausdrücke von Verachtung ihr
Ohr trafen, mit welchen die tugendreichen Dienstmädchen sehr freigebig
waren. Ihre Blässe nahm zu, als der Bediente wieder herunterkam und ihr
sagte, daß sie hinaufgehen könne.

«Rechtlich sein hilft zu nichts in dieser Welt», bemerkte das erste
Dienstmädchen.

«Messing hat's besser als das Gold, das die Feuerprobe bestanden hat»,
sagte das zweite.

Das dritte begnügte sich damit, seine Verwunderung darüber
auszusprechen: «aus welchem besseren Stoffe die Damen wohl sein
möchten»; und das vierte übernahm die Sopranstimme im Quartett: «'s ist
'ne Schande», womit die vier Dianen schlossen.

Ohne auf dieses alles zu achten -- denn sie hatte wichtigere Dinge
auf dem Herzen -- folgte Nancy mit Beben dem Bedienten in ein kleines
Vorzimmer, das durch eine von der Decke herabhängende Lampe erleuchtet
war, und in welchem ihr Führer sie allein ließ.




40. Kapitel.

    Eine seltsame Zusammenkunft, die eine Folge von den im vorigen
    Kapitel erzählten Ereignissen ist.


Nancy hatte ihr ganzes Leben in den Straßen und den ekelhaftesten
Höhlen des Lasters der Hauptstadt zugebracht, dabei aber immer noch
sich einen Rest von der Natur des Weibes bewahrt; und als sie die
leichten, der Tür sich nähernden Schritte vernahm und des weiten
Abstandes der Personen gedachte, die das Gemach im nächsten Augenblick
einschließen würde, fühlte sie sich durch die Last ihrer tiefen Schmach
gänzlich zu Boden gedrückt und fuhr in sich zusammen, wie wenn sie
die Gegenwart der Dame kaum zu ertragen vermöchte, bei welcher sie
vorgelassen zu werden gebeten hatte.

Allein gegen die besseren Gefühle kämpfte der Stolz an -- die Sünde
der Niedrigsten und Verworfensten wie der Höchststehenden und im
Guten befestigt sich Dünkenden. Die elende Genossin von Dieben und
Bösewichtern aller Art, die tiefgesunkene Bewohnerin der gemeinsten
Schlupfwinkel, die Genossin der Auswürflinge der Gefängnisse und der
Galeeren, die selbst im Galgenbereiche Lebende -- selbst diese mit
Schmach und Schande Beladene empfand zu viel Stolz, um auch nur einen
schwachen Schimmer des weiblichen Gefühls zu verraten, welches ihr als
Schwäche erschien, während es noch das einzige Band war zwischen ihr
und der besseren Menschheit, deren äußere Spuren und Kennzeichen alle
ihr wüstes Leben bei ihr vertilgt hatte.

Sie erhob die Augen zur Genüge, um zu gewahren, daß die Gestalt, welche
jetzt erschien, die eines zartgebauten, holden Mädchens war; sie senkte
die Blicke nieder und sagte, den Kopf mit angenommener Gleichgültigkeit
emporwerfend: «Es hat schwer gehalten, zu Ihnen gelassen zu werden,
Lady. Wär' ich empfindlich gewesen und fortgegangen, wie es viele getan
haben würden, Sie möchten es dereinst bereut haben, und nicht ohne
Grund.»

«Es tut mir leid, wenn man Sie unartig behandelt hat», erwiderte Rose.
«Denken Sie nicht mehr daran und sagen Sie mir, weshalb Sie mich zu
sprechen wünschen.»

Der gütige Ton, in welchem sie antwortete, ihre freundlich klingende
Stimme, ihr sanftes Wesen und der Umstand, daß sie gar keinen Hochmut,
kein Mißfallen zeigte, überraschten Nancy dergestalt, daß sie in einen
Tränenstrom ausbrach.

«O Lady, Lady!» rief sie, die aufgehobenen Hände leidenschaftlich
ringend, «wenn mehrere Ihresgleichen wären, würden weniger
meinesgleichen sein -- gewiß -- gewiß!»

«Setzen Sie sich», sagte Rose; «Ihre Worte gehen mir in der Tat an das
Herz. Wenn Sie in bedürftiger Lage oder sonst unglücklich sind, so werde
ich mich glücklich schätzen, Ihnen, wenn ich es vermag, beizustehen --
glauben Sie es mir. Setzen Sie sich.»

«Lassen Sie mich nur stehen, Lady,» sagte das Mädchen, noch immer
Tränen vergießend, «und reden Sie nicht so gütig zu mir, bis Sie
mich besser kennen lernen. Doch es wird spät. Ist -- ist -- jene Tür
verschlossen?»

«Ja», erwiderte Rose, einige Schritte zurückweichend, als ob sie im
Notfalle der Hilfe nahe zu sein wünschte. «Weshalb aber?»

«Weil ich im Begriff bin, mein Leben und das Leben anderer in Ihre
Hände zu legen. Ich bin das Mädchen, das den kleinen Oliver zu Fagin,
dem alten Juden, an jenem Abend wieder zurückschleppte, als er das Haus
in Pentonville verließ.»

«Sie!» sagte Rose Maylie.

«Ja, ich, Lady. Ich bin die Schändliche, von der Sie ohne Zweifel
gehört haben, die unter Dieben lebt und die, Gott helfe mir! solange
ich zurückdenken kann, kein besseres Leben oder freundlichere Worte,
als meine Genossen mir geben, gekannt hat. Ja, weichen Sie nur
immerhin entsetzt vor mir zurück, Lady. Ich bin jünger, als Sie nach
meinem Aussehen glauben mögen: allein ich bin daran gewöhnt, und die
ärmsten Frauen entziehen sich meiner Berührung, wenn ich durch die
dichtgedrängten Straßen gehe.»

«Wie schrecklich!» sagte Rose, sich von dem Mädchen unwillkürlich noch
weiter entfernend.

«Danken Sie auf Ihren Knien dem Himmel, geehrte Lady,» rief die
Unglückliche aus, «daß Sie Angehörige haben, die Sie in Ihrer Jugend
bewacht und gepflegt, und daß Sie niemals wie ich seit der frühesten
Kindheit, von Kälte und Hunger, von Völlerei und Trunkenheit und -- und
von noch etwas viel Schlimmerem, als dieses alles ist, umgeben gewesen
sind. Ich darf es sagen, denn elende Gassen und wüste Höhlen sind meine
Behausung gewesen und werden mein Sterbebett sein.»

«Ich bemitleide Sie!» sagte Rose mit bebender Stimme. «Es ist ja
herzzerreißend, Sie anzuhören.»

«Gottes Segen über Sie und Ihre Güte!» erwiderte das Mädchen. «Wenn
Sie wüßten, wie es mir bisweilen zumute ist, Sie würden mich bedauern,
glauben Sie mir. Doch ich habe mich fortgeschlichen von Leuten, die
mich sicherlich ermorden würden, wüßten sie, daß ich hier gewesen
bin, um Sie von Dingen, die ich ihnen abgehorcht habe, in Kenntnis zu
setzen. Ist Ihnen ein Mensch namens Monks bekannt?» Rose verneinte.

«Er kennt Sie», fuhr das Mädchen fort, «und wußte, daß Sie hier
wohnten, denn nur dadurch, daß er es einem anderen sagte, ward es mir
möglich, Sie aufzufinden.»

«Ich habe den Namen niemals nennen hören», sagte Rose.

«Nun, so führt er unter uns einen anderen, was ich wohl schon früher
vermutet habe. Vor einiger Zeit und bald nachdem Oliver in der Nacht
des beabsichtigten Raubes in Ihr Haus gehoben wurde, behorchte ich
diesen Menschen, auf welchen ich Verdacht geworfen, als er mit Fagin
eine Unterredung hatte. Ich erfuhr bei der Gelegenheit, daß Monks --
der Mann, nach welchem ich Sie vorhin fragte --»

«Wohl, ich verstehe schon», sagte Rose.

«Daß Monks den Knaben an eben dem Tage, als wir ihn verloren, mit zwei
von unseren Knaben zufällig erblickte und sogleich in ihm das Kind
erkannt hatte, welchem er auflauerte, wiewohl ich mir nicht erklären
konnte, weshalb. Er wurde mit Fagin darüber einig, daß der Jude,
falls Oliver wieder zurückgebracht würde, eine gewisse Summe und noch
mehr erhalten solle, wenn er einen Dieb aus ihm machte, was Monks zu
irgendeinem Zweck wünschte.»

«Zu welchem Zwecke?» fragte Rose.

«Als ich horchte, um es zu erlauschen, erblickte er meinen Schatten an
der Wand,» fuhr das Mädchen fort, «und es gibt außer mir nicht sehr
viele Menschen, die, um der Entdeckung zu entgehen, zeitig genug sich
aus dem Hause gefunden hätten. Mir gelang es indes, und ich sah ihn
erst am gestrigen Abende wieder.»

«Und was trug sich da zu?»

«Ich will es Ihnen sagen, Lady. Er kam gestern wieder zu Fagin. Sie
gingen wieder die Treppe hinauf; ich versteckte mich und hüllte mich so
ein, daß mich mein Schatten nicht verraten konnte, und horchte abermals
an der Tür. Die ersten Worte, die ich Monks sagen hörte, waren diese:
>So liegen denn die einzigen Beweise, daß der Oliver der Knabe ist, auf
dem Grunde des Stromes, und die alte Hexe, die sie von seiner Mutter
erhielt, verfault in ihrem Sarge.< Sie lachten und sprachen von der
glücklichen Ausführung des Streichs, und Monks, der noch weiter von
dem Knaben sprach und sehr ingrimmig wurde, sagte: obwohl er des jungen
Teufels Geld jetzt sicher genug hätte, so würde er es doch lieber auf
andere Art erlangt haben; denn welch eine Lust würde es sein, das
prahlerische Testament des Vaters dadurch über den Haufen zu werfen,
daß man den Knaben durch alle Gefängnisse der Hauptstadt hetzte und ihn
dann wegen eines todeswürdigen Verbrechens vor Gericht zöge, was Fagin
leicht würde veranstalten können, nachdem er ihn obendrein mit großem
Vorteile benutzt haben würde.»

«Was ist das alles?» rief Rose entsetzt aus.

«Die Wahrheit, Lady, obwohl es von meinen Lippen kommt», versetzte
das Mädchen. «Dann sagte er unter Verwünschungen, die für mein Ohr
gewöhnlich genug sind, den Ihrigen aber fremd und schauerlich klingen
müßten, er würde es tun, wenn er seinen Haß ohne Gefahr für seinen
eigenen Hals dadurch befriedigen könnte, daß er dem Knaben das
Leben nähme; es dürfte aber zu gefährlich sein; er würde ihm jedoch
überall im Leben auflauern und könnte, wenn er sich die Geburt und
Lebensgeschichte des Knaben zunutze machte, ihm dennoch Schaden genug
zufügen. >Kurzum, Fagin,< sagte er, >Jude, der du bist, du hast noch
nie Fallstricke gelegt, wie ich sie zum Verderben meines jungen Bruders
legen werde.<»

«Sein Bruder!» rief Rose bestürzt.

«Das waren seine Worte», sagte Nancy, sich besorgt umschauend, wie sie
es fast unablässig getan hatte, denn Sikes' finstere Gestalt schwebte
ihr beständig vor der Seele.

«Und mehr noch. Indem er von Ihnen und der anderen Dame sprach, äußerte
er, der Himmel oder der Teufel müsse wider ihn gewesen sein, als Oliver
in Ihre Hände geraten sei, und sagte mit Hohngelächter: darin läge
ebenfalls einiger Trost. Denn wieviel tausend und hunderttausend Pfund
würden Sie nicht geben, wenn Sie sie hätten, zu erfahren, wer Ihr
zweibeiniger Schoßhund wäre.»

«Sie wollen doch nicht sagen, daß das alles ernstlich gemeint war»,
sagte Rose erblassend.

«Wenn jemals ein Mensch im Ernst gesprochen, so tat ich es in diesen
Augenblicken», erwiderte das Mädchen, traurig den Kopf schüttelnd; «und
auch er pflegt nicht zu scherzen, wenn sein Haß in ihm lebendig ist.
Ich kenne viele, die noch Schlimmeres üben, aber ich würde sie alle
lieber zehnmal als jenen Monks ein einziges Mal darüber sprechen hören.
Doch es wird spät, und ich muß nach Hause zurückkehren, um ja nicht
den Verdacht aufkommen zu lassen, daß ich zu einem solchen Zweck hier
gewesen wäre. Ich muß nach Hause zurückeilen.»

«Doch, was kann ich tun?» fragte Rose. «Welchen Nutzen kann ich ohne
Sie aus Ihrer Mitteilung ziehen? Zurückkehren wollen Sie! Wie können
Sie zu Genossen zurückzukehren wünschen, die Sie mit so schrecklichen
Farben schildern? Wenn Sie Ihre Aussage in Gegenwart eines Herrn,
welchen ich augenblicklich herbeirufen kann, wiederholen wollen, so
können Sie binnen einer halben Stunde an einen sicheren Ort gebracht
werden.»

«Ich wünsche aber zurückzukehren», versetzte das Mädchen. «Ich muß
zurückkehren, weil -- ach, wie kann ich mit einer unschuldigen Dame,
wie Sie sind, über solche Dinge reden? -- weil unter den Männern, von
welchen ich Ihnen erzählt habe, sich einer befindet, der Schrecklichste
von allen, den ich nicht zu verlassen vermag; nein -- und wenn ich auch
dadurch von dem ruchlosen, fürchterlichen Leben erlöst werden könnte,
das ich jetzt führe!»

«Daß Sie zugunsten des teuren Knaben sich schon einmal bemüht haben;
daß Sie unter so großer Gefahr hierher gekommen sind, um das, was Sie
gehört, mir zu enthüllen; Ihre Mienen, die mich von der Wahrheit Ihrer
Angaben überzeugen; Ihre offenbare Reue und Ihr Schamgefühl: alles
berechtigt mich dazu, zu glauben, daß Sie wieder auf den rechten Weg
gebracht werden können. Oh», fuhr die tiefbewegte Rose Maylie, die
Hände faltend, während Tränen über ihre Wangen hinabliefen, fort,
«hören Sie auf das Flehen einer Angehörigen Ihres eigenen Geschlechts,
der ersten -- gewiß der ersten, die jemals mit der Stimme des Mitleids
und der Bangigkeit um Ihr Seelenheil zu Ihnen geredet hat. Hören Sie
auf meine Worte und lassen Sie sich durch mich zu einem besseren Dasein
erretten!»

«Lady,» versetzte das Mädchen, auf die Knie sinkend, «teure,
engelgleiche Lady, ja, Sie sind die erste, die mich jemals durch Worte,
wie diese sind, beseligt hat, und hätte ich sie vor Jahren vernommen,
so hätten sie mich einem sündhaften und leidvollen Leben entreißen
können; doch es ist zu spät -- zu spät.»

«Zur Reue und Buße ist es niemals zu spät», entgegnete Rose.

«Es ist dennoch zu spät!» rief Nancy in einem Tone aus, der ihre ganze
Seelenqual verriet. «Ich kann ihn jetzt nicht mehr verlassen -- ich
vermöchte es nicht, seinen Tod herbeizuführen.»

«Und weshalb sollten Sie es?» fragte Rose.

«Nichts könnte ihn retten», jammerte das Mädchen. «Wenn ich anderen
erzählte, was ich Ihnen offenbart habe, und dadurch seine Verhaftung
veranlaßte, er müßte ohne Rettung sterben. Er ist der verwegenste von
allen, und hat so entsetzliche Dinge begangen!»

«Ist es möglich,» rief Rose, «daß Sie einem solchen Menschen zuliebe
jeder Hoffnung auf die Zukunft und der Gewißheit der Rettung für die
Gegenwart entsagen können? Es ist Wahnsinn.»

«Ich weiß nicht, was es ist,» versetzte das Mädchen, «ich weiß nur,
daß es so ist, und nicht allein bei mir, sondern bei Hunderten, die
ebenso schlecht und elend sind, wie ich es bin. Ich muß zurück. Ob es
der Zorn Gottes ist, wegen des vielen Bösen, das ich begangen habe,
weiß ich nicht; aber ich fühle mich trotz aller Leiden und aller harten
Behandlung unwiderstehlich zu ihm hingezogen, was, glaub' ich, auch
dann der Fall sein würde, wenn ich überzeugt wäre, daß ich noch durch
seine Hand sterben müßte.»

«Was soll ich tun?» fragte Rose. «Ich müßte Sie eigentlich nicht
fortlassen.»

«Ja, ja, Lady, Sie werden es», entgegnete das Mädchen. «Sie werden mein
Fortgehen nicht hindern, weil ich in Ihre Güte Vertrauen gesetzt und
Ihnen, wie ich es hätte tun können, kein Versprechen abgerungen habe.»

«Wozu nützt denn aber Ihre Mitteilung?» beharrte Rose. «Dies Geheimnis
muß enthüllt werden; welcher Vorteil kann sonst für Oliver, dem zu
dienen Ihnen so sehr am Herzen liegt, daraus erwachsen, daß Sie es mir
anvertraut haben?»

«Sie werden sicher irgendeinen wohlwollenden Herrn kennen, dem Sie es
mitteilen können, und der Ihnen Rat erteilen wird», erwiderte Nancy.

«Doch, wo finde ich Sie, wenn ich Ihrer bedürfen sollte?» fragte Rose.
«Ich will nicht fragen, wo jene fürchterlichen Menschen wohnen, allein,
wo wird man Sie an irgendeinem zu bestimmenden Tage wiedersehen können?»

«Versprechen Sie mir, daß mein Geheimnis auf das strengste bewahrt
werden soll, und daß Sie allein oder doch nur mit dem Manne kommen,
dem Sie es anvertrauen wollen, und daß man mir weder auflauere noch
nachfolge?»

«Ich verspreche es feierlichst», erwiderte Rose.

«Wohlan, so will ich jeden Sonntag von elf bis zwölf Uhr abends, wenn
ich am Leben bleibe, auf der Londoner Brücke auf und nieder gehen»,
verhieß Nancy unbedenklich.

«Warten Sie noch einen Augenblick», sagte Rose, Nancy, die schon nach
der Tür eilte, zurückhaltend. «Erwägen Sie noch einmal Ihre Lage und
die Gelegenheit, die Ihnen geboten wird, sich derselben zu entreißen.
Sie haben nicht allein als freiwillige Überbringerin einer so wichtigen
Kunde, sondern auch als eine fast unwiederbringlich Verlorene Ansprüche
auf meinen Beistand. Wollen Sie in der Tat zu der Räuberbande und dem
schrecklichen Manne zurückkehren, da doch ein einziges Wort Sie retten
kann? Was für ein Zauber ist es, der Sie unwiderstehlich zurückzuziehen
und der Gottlosigkeit und dem Elend preiszugeben vermag? Ach, befindet
sich denn in Ihrem Herzen keine Saite, die ich zu berühren vermöchte
-- regt sich kein Gefühl in ihm, das gegen diese Verblendung ankämpfen
könnte?»

«Wenn Damen, so jung, so freundlich und schön, wie Sie sind, ihre
Herzen verschenken,» versetzte das Mädchen mit fester Stimme, «so
macht die Liebe sie zu allem fähig -- selbst Ihresgleichen, die Sie
eine Heimat, Angehörige, Freunde, zahlreiche Bewunderer, alles haben,
was Ihr Herz ausfüllen kann. Wenn Frauen wie ich, die wir kein Dach
als den Sargdeckel, in Krankheit und Tod keinen Beistand als die
Krankenwärterin des Hospitals haben, einem Manne unser angefaultes
Herz hingeben und ihn die Stelle ausfüllen lassen, die einst von den
Eltern, der Heimat und den Freunden ausgefüllt wurde, oder die unser
ganzes elendes Leben hindurch eine leere und wüste Stätte gewesen ist:
wer kann hoffen uns zu heilen? Bemitleiden Sie uns, Lady -- bemitleiden
Sie uns darum, daß uns nur ein weibliches Gefühl geblieben ist, und daß
dieses Gefühl, durch die schwere Ahndung des Himmels, statt unser Trost
und Stolz zu sein, zu einem Fluche und die Quelle neuer Leiden und
Mißhandlungen wird.»

«Sie werden doch eine Kleinigkeit von mir annehmen,» sagte Rose nach
einer Pause, «die Sie in den Stand setzen wird, ohne Schande zu leben
-- wenigstens bis wir uns wiedersehen?»

«Keinen Heller», erwiderte das Mädchen, mit der Hand abwehrend.

«Verschließen Sie Ihr Herz doch nicht gegen meine Anerbietungen, Ihnen
Beistand zu leisten», sagte Rose, ihr näher tretend. «Gewiß, ich
wünsche Ihnen nützlich zu sein.»

«Sie würden mir am nützlichsten sein, Lady, wenn Sie mir mit einem
Male das Leben nehmen könnten», versetzte Nancy händeringend; «denn
der Gedanke an das, was ich bin, hat mir in dieser kurzen Stunde ein
schwereres Herzweh verursacht, als ich jemals empfunden habe, und es
würde ein Gewinn sein, nicht in der Hölle zu sterben, in der ich gelebt
habe. Gottes Segen über Sie, süße Lady, und möge der Himmel ebensoviel
Glück auf Ihr Haupt herabsenden, wie ich auf das meine Schande geladen
habe!»

Mit diesen Worten und unter lautem Schluchzen verließ die
Bejammernswerte das Zimmer, während Rose, durch die eben beendete
Unterredung, die mehr einem flüchtigen Traume als der Wirklichkeit
ähnlich sah, fast überwältigt auf einen Stuhl niedersank und ihre
verworrenen Gedanken zu sammeln suchte.




41. Kapitel.

    Welches neue Entdeckungen enthält und zeigt, daß Überraschungen,
    gleich Unglücksfällen, selten allein kommen.


Roses Lage war in der Tat nicht wenig schwierig, denn während sie das
lebhafte Verlangen empfand, das geheimnisvolle Dunkel zu durchdringen,
das Olivers Geschichte umhüllte, konnte sie doch nicht umhin, das
Vertrauen zu ehren, welches die Unglückliche, mit der sie soeben
gesprochen, in sie, als ein junges, argloses Mädchen, gesetzt hatte.
Die Worte und das ganze Wesen derselben hatten Rose tief gerührt, und
ihrer Zuneigung für ihren jugendlichen Schützling gesellte sich der
ebenso heiße Wunsch hinzu, das Gefühl der Reue in der Verlorenen zu
erwecken und ihr neue Lebenshoffnung einzuflößen.

Mrs. Maylie hatte beabsichtigt, nur drei Tage in London zu verweilen
und dann auf einige Wochen nach einem entfernten Ort an der Seeküste
abzureisen. Es war Mitternacht zwischen dem ersten und zweiten Tage.
Für welche Schritte konnte sich Rose binnen achtundvierzig Stunden
entscheiden? und wie konnte sie die Reise aufzuschieben suchen, ohne
Vermutungen zu wecken, daß sich etwas Besonderes ereignet hätte?

Mr. Losberne weilte im Hause und beabsichtigte, auch noch die
beiden folgenden Tage zu bleiben; allein Rose kannte die ungestüme
Lebhaftigkeit des Ehrenmannes zu wohl und dachte sich im voraus zu
deutlich den Zorn, welchen er im ersten Ausbruch seiner Entrüstung
auf das Werkzeug der zweiten Entführung Olivers werfen würde, um es
zu unternehmen, ihm das Geheimnis, solange ihre Gegenvorstellungen
zugunsten Nancys von keiner Seite unterstützt wurden, anzuvertrauen.
Dies waren die Gründe, welche Rose zu dem Entschlusse bewogen, ihre
Tante nur mit der größten Vorsicht von der Sache in Kenntnis zu setzen,
da dieselbe, wie sie voraussah, nicht verfehlen würde, sich mit dem
würdigen Doktor über die Angelegenheit zu beraten. Aus denselben
Gründen war nicht daran zu denken, sich an einen Rechtskundigen zu
wenden, selbst wenn sie gewußt, wie sie sich dabei zu benehmen hätte.
Einmal stieg der Gedanke in ihr auf, Harrys Beistand in Anspruch zu
nehmen; allein er weckte die Erinnerung an ihr letztes Scheiden von ihm
wieder auf, und es erschien ihr unwürdig, ihn wieder in ihre Nähe zu
ziehen, da es ihm -- und bei dieser Vorstellung traten ihr die Tränen
in die Augen -- jetzt vielleicht gelungen war, sie zu vergessen und auf
eine andere Art glücklich zu sein.

Durch diese wechselnden Betrachtungen aufgeregt und sich bald für
diese, bald für jene Maßregel entscheidend, bald alle verwerfend,
brachte Rose die Nacht in schlafloser Bangigkeit hin und faßte,
nachdem sie am folgenden Tage die Sache abermals überlegt hatte, den
verzweifelten Entschluß, trotz aller Bedenken Harry Maylie zu Rate zu
ziehen.

«Wenn es ihm auch peinlich sein wird, hierher zurückzukehren, ach! wie
peinlich wird es für mich sein!» dachte sie sinnend. «Doch vielleicht
kommt er nicht; er wird vielleicht schriftlich antworten oder auch
kommen und mich ängstlich zu meiden suchen -- wie damals, als er
fortreiste. Ich hatte es nicht erwartet; doch es war für uns beide
besser -- viel besser»; und Rose legte hier die Feder nieder und wandte
sich hinweg, gleichsam, als ob sie zu vermeiden wünschte, daß auch
nur das Papier, das ihre Botschaft ausrichten sollte, ein Zeuge ihrer
Tränen wäre.

Sie hatte die Feder schon zwanzigmal wieder ergriffen und sie
ebensooft wieder zur Seite gelegt und die Fassung der allerersten
Zeile ihres Schreibens hin und her überlegt, ohne auch nur eine Silbe
niedergeschrieben zu haben, als Oliver, der mit Mr. Giles von einer
Wanderung durch die Stadt zurückgekehrt war, in atemloser Hast und
lebhafter Unruhe in das Zimmer trat, wie wenn ein neues Unglück zu
fürchten wäre.

«Oliver, warum siehst du so erschreckt aus?» fragte Rose, ihm
entgegentretend. «Rede, mein Kind!»

«Ich kann kaum; mir ist es, als ob ich ersticken müßte», erwiderte
der Knabe. «Ach! daß ich ihn doch noch gesehen habe, und daß Sie sich
überzeugen werden, daß ich Ihnen die reine Wahrheit erzählt habe!»

«Ich habe nie daran gezweifelt, daß du die Wahrheit gesprochen hast,
mein Liebling», versetzte Rose besänftigend. «Doch was bedeutet dies
alles -- von wem ist die Rede?»

«Ich habe den Herrn gesehen,» erwiderte der Knabe, «den Herrn, der so
gütig gegen mich war -- Mr. Brownlow, von dem wir so oft gesprochen
haben.»

«Wo?» fragte Rose.

«Er stieg eben aus einem Wagen und ging in ein Haus», erwiderte Oliver,
indem Freudentränen aus seinen Augen hervorstürzten. «Ich redete ihn
nicht an -- ich konnte nicht, denn er sah mich nicht, und ich zitterte
so, daß ich nicht imstande war, zu ihm zu gehen. Aber Giles erkundigte
sich, ob er in dem Hause wohnte, und man sagte ja. Sehen Sie,» fuhr er
fort, ein Stück Papier entfaltend, «hier steht es; da wohnt er -- ich
will sogleich hingehen. O Gott! ich werde mich nicht fassen können,
wenn ich ihn sehe und seine Stimme wieder höre!»

Rose Maylie hatte unter diesen und noch vielen ähnlichen Ausrufen
der Freude des Knaben große Mühe, Mr. Brownlows Adresse zu lesen,
>Craven Street, Strand<; sie beschloß indes nach einiger Zeit, Olivers
Entdeckung ohne Säumen zu benutzen.

«Schnell!» sagte sie, «gib Befehl, einen Mietwagen kommen zu lassen,
und halte dich bereit, mich zu begleiten. Ich werde dich, ohne einen
Augenblick zu verlieren, hinbringen und will nur erst meiner Tante
sagen, daß wir auf eine Stunde ausfahren wollen; ich werde ebenso rasch
fertig sein wie du selbst.»

Es bedurfte bei Oliver keiner Mahnung zur Eile, und in weniger als fünf
Minuten befanden sie sich auf dem Wege nach der bezeichneten Straße.
Als sie angelangt waren, ließ Rose Oliver unter dem Vorwande, den alten
Herrn auf sein Erscheinen vorzubereiten, allein im Wagen, stieg aus
und schickte durch den Diener ihre Karte mit der Bitte hinauf, Mr.
Brownlow in sehr dringenden Angelegenheiten sprechen zu dürfen. Der
Diener kehrte bald wieder zurück, um sie zu ersuchen, hinaufzukommen.
Sie folgte ihm in eins der oberen Zimmer, wo sie einen ältlichen, in
einem dunkelgrünen Rocke sich präsentierenden Herrn, in dessen Mienen
unverkennbare Herzensgüte sich ausdrückte, fand. Nicht weit von ihm
erblickte sie einen zweiten alten Herrn in Nankingbeinkleidern und
Gamaschen, der nicht besonders wohlwollend aussah und dasaß, die Hände
auf den Knauf eines schweren Spazierstocks gestützt und das Kinn auf
demselben ruhend lassend.

«Ah!» sagte der Herr im grünen Rocke, eilfertig und mit Zuvorkommenheit
aufspringend, «ah, entschuldigen Sie, mein gnädiges Fräulein -- ich
glaubte, es wäre irgendeine zudringliche Person, die -- Sie werden mich
gütigst entschuldigen. Bitte, nehmen Sie Platz!»

«Mr. Brownlow, wenn ich nicht irre, Sir?» sagte Rose, nachdem sie auf
den andern Herrn einen Blick geworfen hatte.

«So ist mein Name, ja», erwiderte der alte Herr. «Dies ist mein Freund,
Mr. Grimwig. Grimwig, Sie haben wohl die Gefälligkeit und verlassen uns
auf einige Minuten.»

«Ich glaube nicht, daß es notwendig sein wird, den Herrn zu bemühen»,
bemerkte Rose. «Wenn ich nicht irre, so ist ihm die Angelegenheit, in
welcher ich Sie zu sprechen wünsche, nicht fremd.»

Brownlow gab seine Einwilligung durch eine leichte Kopfneigung zu
erkennen, und Grimwig, der eine sehr steife Verbeugung gemacht hatte
und aufgestanden war, machte eine zweite sehr steife Verbeugung und
nahm wieder Platz.

«Was ich Ihnen mitzuteilen habe, wird Sie ohne Zweifel sehr
überraschen», begann Rose etwas verlegen. «Sie erwiesen einst einem
mir sehr teuern jungen Freunde viel Wohlwollen und Güte, und ich bin
überzeugt, daß es Sie freuen wird, wieder von ihm zu hören.»

«Einem jungen Freunde!» sagte Mr. Brownlow. «Darf ich seinen Namen
wissen?»

«Oliver Twist!» erwiderte Rose.

Kaum waren die Worte ihrem Munde entflohen, als Grimwig, der sich
gestellt hatte, als ob ihn der Inhalt eines auf dem Tisch liegenden
großen Buches lebhaft interessierte, dasselbe mit großem Geräusche
zuschlug, sich zurücklehnte, wobei sein Antlitz den Ausdruck des
äußersten Erstaunens annahm, und lange mit großen, stieren Augen dasaß,
worauf er, als ob er sich schämte, so viel innere Bewegung an den
Tag gelegt zu haben, sich in seine vorige Stellung gleichsam wieder
zurückschnellte und, indem er gerade vor sich hinstarrte, einen langen,
pfeifenden Ton erschallen ließ, der nicht in der leeren Luft, sondern
in den innersten Höhlen seines Magens zu ersterben schien.

Mr. Brownlow war nicht weniger erstaunt, wiewohl sein Erstaunen sich
auf eine weit minder seltsame Art kundgab. Er rückte seinen Stuhl näher
zu Rose heran und sagte: «Erzeigen Sie mir den Gefallen, mein liebes,
gnädiges Fräulein, die Güte und das Wohlwollen, von welchem Sie reden,
und wovon, Sie ausgenommen, niemand weiß, gänzlich außer Frage zu
lassen; und wenn Sie irgend Beweise herbeizubringen vermögen, welche
geeignet sind, mir die üble Meinung zu benehmen, die ich vormals von
dem genannten unglücklichen Kinde zu hegen mich bewogen fand, so teilen
Sie sie mir mit -- ich bitte dringend darum.»

«Ein böser Bube -- ich will meinen Kopf aufessen, wenn er es nicht
ist», brummte Grimwig in sich hinein wie ein Bauchredner und ohne einen
Gesichtsmuskel in Bewegung zu setzen.

«Der Knabe besitzt einen reinen Sinn und ein warmes Herz», sagte Rose,
vor Unmut errötend; «und die Allmacht, der es gefallen, ihm Prüfungen,
die über seine Jahre hinausgingen, aufzuerlegen, hat in seiner Brust
Gefühle und Gesinnungen keimen lassen, welche unzähligen Ehre machen
würden, die seine Jahre sechsfach zählen.»

«Ich bin erst einundsechzig,» bemerkte Grimwig mit derselben starren
Unbeweglichkeit, «und da es mit dem Teufel zugehen müßte, wenn dieser
Oliver nicht wenigstens zwölf Jahr alt ist, so sehe ich das Zutreffende
der Bemerkung nicht ein.»

«Achten Sie nicht auf meinen Freund, Miß Maylie», sagte Brownlow; «er
meint es doch nicht so.»

«Das tut er allerdings», brummte Grimwig vor sich hin.

«Nein, er tut es nicht», beharrte Brownlow, der offenbar immer
erzürnter wurde.

«Er will seinen Kopf aufessen, wenn er es nicht tut», beteuerte Grimwig.

«Er verdiente, ihn zu verlieren, wenn er es täte», entgegnete Brownlow.

«Und er möchte denjenigen sehen, der es zu versuchen wagte, ihm den
Kopf zu nehmen», erwiderte Grimwig, seinen Stock mit Heftigkeit gegen
den Fußboden stoßend.

Nachdem die alten Herren soweit gediehen waren, nahmen beide eine
Prise Schnupftabak und drückten darauf, gemäß ihrer unabänderlichen
Gewohnheit, einander die Hände.

«Und nun, Miß Maylie,» begann Brownlow, sich wieder zu Rose wendend,
«lassen Sie uns zu dem Gegenstande zurückkehren, an welchem Ihre
Menschenliebe einen so großen Anteil nimmt. Darf ich wissen, welche
Kunde Sie von dem armen Knaben besitzen? Erlauben Sie mir, Ihnen vorher
mitzuteilen, daß ich, um ihn ausfindig zu machen, alle mir zu Gebote
stehenden Mittel erschöpft habe, und daß seit meiner Reise außer Landes
meine erste Meinung, daß er mich belogen und durch seine ehemaligen
Genossen beredet gewesen, mich zu bestehlen, bedeutend erschüttert
worden ist.»

Rose, welcher diese Rede Zeit gelassen hatte, ihre Gedanken zu sammeln,
berichtete nun Brownlow alles, was sich mit Oliver zugetragen, seit er
das Haus desselben verlassen, und verschwieg ihm einstweilen nur Nancys
unter vier Augen ihm anzuvertrauende Mitteilungen. Sie schloß mit
der Versicherung, daß der einzige Kummer, den der Knabe seit einigen
Monaten empfunden, dem Umstande zuzuschreiben sei, daß er seinen
ehemaligen Wohltäter und Freund nirgends habe finden können.

«Gott sei Dank!» rief der alte Herr. «Diese Nachricht macht mich
glücklich, sehr glücklich. Doch Sie haben mir nicht gesagt, wo er sich
gegenwärtig befindet, Miß Maylie. Verzeihen Sie mir -- doch weshalb
haben Sie ihn nicht mitgebracht?»

«Er wartet im Wagen vor der Tür», erwiderte Rose.

«Vor meiner Tür?» rief der alte Herr freudig überrascht aus, eilte,
ohne ein Wort zu sagen, hinaus, die Treppe hinunter, sprang auf den
Wagentritt und in den Wagen hinein.

Sobald er fort war, hob Grimwig den Kopf empor, balancierte seinen
Stuhl auf einem Hinterbeine und beschrieb, ohne aufzustehen und mit
Hilfe seines Stockes und des Tisches, drei ganze Kreise. Nachdem er die
Evolution glücklich ausgeführt hatte, sprang er auf und humpelte nach
besten Kräften zum wenigsten ein dutzendmal im Zimmer auf und ab, blieb
plötzlich vor Rose stehen und küßte sie ohne alle weitere Einleitung.

«Pst!» sagte er, als Rose, über dieses ungewöhnliche Verfahren ein
wenig erschreckt, aufstand; «seien Sie ohne Furcht. Ich bin alt genug,
um Ihr Großvater zu sein. Sie sind ein wackeres, ein sehr gutes
Mädchen -- ich habe Sie lieb. Da kommen sie!»

Bei diesen Worten warf er sich mit einer geschickten Wendung auf seinen
Sitz, und in demselben Augenblick trat Brownlow mit Oliver herein, den
Grimwig sehr gnädig begrüßte. Ach, wenn die Freude dieses Augenblicks
Roses einzige Belohnung gewesen wäre für alle ihre Sorge und Angst um
Oliver, sie würde sich hinlänglich belohnt gefühlt haben.

«Es ist aber noch jemand, den wir nicht vergessen dürfen», sagte
Brownlow, nach der Klingelschnur greifend.

«Sage Mrs. Bedwin, sie möchte einmal heraufkommen», befahl er dem
hereintretenden Diener.

Die bejahrte Haushälterin erschien sogleich, machte ihren Knicks und
blieb, des Befehls des Herrn gewärtig, an der Tür stehen.

«Mein Gott, Sie werden ja alle Tage blinder», sagte Brownlow ein wenig
verdrießlich.

«Mag wohl sein, Sir», erwiderte die gute Alte. «In meinen Jahren
pflegen die Augen nicht schärfer zu werden, Sir.»

«Das hätte ich Ihnen auch sagen können», entgegnete Brownlow. «Doch
setzen Sie Ihre Brille auf und sehen Sie zu, ob Sie nicht selbst
entdecken können, weshalb ich Sie habe heraufkommen lassen.»

Die alte Frau begann sogleich in ihren Taschen zu wühlen; aber Olivers
Geduld hielt die Probe nicht aus, er überließ sich dem Drange seiner
Gefühle und warf sich in ihre Arme.

«Gott sei mir gnädig! -- es ist mein unschuldiger Knabe!» rief sie aus,
indem sie ihn zärtlich in die Arme drückte.

«Meine liebe alte Pflegemutter!» rief Oliver.

«Gott, ich wußte es wohl, daß er zurückkehren würde! Wie gesund und
blühend er aussieht, und er ist obendrein wie 'nes Edelmannes Sohn
gekleidet! Wo bist du so lange, so lange gewesen! Ach! es ist dasselbe
süße Gesichtchen, aber nicht so blaß; dasselbe sanfte Auge, aber nicht
so trübe. Sie sind mir gar nicht aus dem Sinn gekommen und ihr stilles
Lächeln auch nicht; ich habe sie Tag für Tag neben meinen lieben
Kindern gesehen, die, seit ich ein glückliches junges Weib war, tot und
dahingegangen sind.»

Sich so ihrer Redseligkeit überlassend und Oliver bald von sich
haltend, um ihn genauer ansehen zu können, und ihn bald zärtlich an die
Brust drückend und ihm die Locken aus dem Gesichte streichend, weinte
und lachte die gute alte Seele in einem Atem.

Brownlow überließ beide dem Austausch ihrer Gefühle und führte Rose
in ein anderes Zimmer, wo sie ihm einen ausführlichen Bericht über
die Unterredung mit Nancy erstattete, die ihn nicht wenig überraschte
und in Verwirrung und Unruhe setzte. Rose teilte ihm auch ihre Gründe
mit, weshalb sie nicht ihren Freund Losberne zunächst zum Vertrauten
gemacht hätte. Der alte Herr äußerte, sie habe daran sehr klug getan,
und erklärte sich bereit, mit dem würdigen Doktor selbst in Beratung zu
treten. Um ihm hierzu eine baldige Gelegenheit zu verschaffen, wurde
verabredet, daß er noch an demselben Abend in der Villa vorsprechen
und daß mittlerweile Mrs. Maylie von allem, was vorgefallen war,
vorsichtig in Kenntnis gesetzt werden sollte. Sobald diese vorläufigen
Bestimmungen getroffen waren, kehrten Rose und Oliver wieder nach Hause
zurück.

Rose hatte das Maß der Entrüstung des trefflichen Doktors keineswegs
überschätzt; denn kaum war ihm Nancys Erzählung mitgeteilt worden,
als er seinen Zorn in einem Strome von Verwünschungen und Drohungen
ergoß, sie zum ersten Schlachtopfer des vereinten Scharfsinnes
der Herrn Blathers und Duff zu machen gelobte und sogar den Hut
aufsetzte, in der Absicht, fortzueilen und den Beistand der genannten
Ehrenmänner in Anspruch zu nehmen. Und er würde im ersten Losstürmen
sein Vorhaben, ohne die Folgen des allergeringsten Nachdenkens zu
würdigen, ohne Zweifel ausgeführt haben, wenn er nicht zurückgehalten
worden wäre, teils durch das ebenso große Ungestüm Brownlows, der
selbst ein reizbares Temperament besaß, und teils durch die Gründe und
Gegenvorstellungen, die man für die zweckdienlichsten erachtete, ihn
von seinem unbesonnenen Verfahren zurückzubringen.

«Was ist aber zum Geier zu tun?» sagte der hitzige Doktor, als sie in
das Zimmer zu den beiden Damen getreten waren. «Wir sollen doch nicht
all das männliche und weibliche Gesindel unseres Dankes versichern und
es bitten, hundert oder ein paar hundert Pfund als ein geringes Zeichen
unserer Achtung und als einen kleinen Beweis unserer Erkenntlichkeit
für ihre Güte gegen Oliver anzunehmen?»

«Das eben nicht», erwiderte Brownlow mit Lachen; «allein wir müssen
besonnen und mit großer Vorsicht handeln.»

«Besonnen und vorsichtig!» rief der Doktor aus. «Ich würde die Halunken
samt und sonders zum --»

«Es ist einerlei, zu wem Sie sie schicken würden», unterbrach ihn
Brownlow. «Doch fragen Sie sich selbst, ob wir, wir mögen sie schicken,
wohin wir wollen, eine Hoffnung haben, dadurch zum Ziele zu gelangen.»

«Zu welchem Ziele?» fragte der Doktor.

«Dem einfachen Ziele, zu erforschen, wer Olivers Eltern gewesen sind,
und ihm seine Erbschaft wieder zuzuwenden, um welche er, sofern alle
vorliegenden Angaben begründet sind, schändlich betrogen worden ist.»

«Ah!» sagte Losberne, sich mit dem Schnupftuche Kühlung zuwehend, «das
hätte ich bald vergessen.»

«Sie begreifen also», fuhr Brownlow fort. «Was würden wir denn Gutes
stiften, wenn wir, angenommen, es wäre ausführbar, ohne die Sicherheit
des armen Mädchens zu gefährden, die Bösewichter dem Arme der
Gerechtigkeit überlieferten?»

«Das Gute, daß einige von ihnen baumelten und die übrigen deportiert
würden», meinte der Doktor.

«Sehr wohl», erwiderte Brownlow lächelnd; «allein sie werden dafür
seinerzeit ohne Zweifel schon selber sorgen, und wenn wir ihnen
vorgreifen, so scheint mir's, wir werden eine arge Don-Quichotterie
begehen und unserm oder doch Olivers Interesse, was aber dasselbe ist,
gerade zuwiderhandeln.»

«Wieso denn?» fragte der Doktor.

«Ist es nicht klar genug,» erwiderte Brownlow, «daß es uns äußerst
schwer werden wird, dem Geheimnisse auf den Grund zu kommen, wenn wir
nicht imstande sind, Monks zum Beichten zu bringen? Das kann aber nur
durch List geschehen und dadurch, daß wir ihn fassen, wenn er eben
nicht von dem übrigen Gelichter umgeben ist. Denn gesetzt auch, daß
er aufgegriffen würde -- wir haben keine Beweise wider ihn. Er hat
(soviel wir wissen, oder so weit es aus den Umständen hervorgeht) an
keinem Diebstahle oder Raube der Bande teilgenommen. Wenn er auch nicht
freigesprochen werden würde, so ist es doch sehr unwahrscheinlich, daß
er eine weitere Strafe erhielte als die, daß man ihn eine Zeitlang als
Landstreicher einsperrte, und sein Mund würde dann hinterher für immer
so fest geschlossen sein, daß wir unsere Zwecke ebensowenig erreichten,
wie wenn er taub, stumm, blind und blödsinnig wäre.»

«Ich frage Sie abermals,» sagte der Doktor heftig, «ob Sie das dem
Mädchen gegebene Versprechen vernünftigerweise für bindend halten --
ein Versprechen, das in der besten und wohlwollendsten Absicht gegeben
ist, aber wirklich --»

«Ich bitte, lassen Sie den Punkt unerörtert, mein verehrtes junges
Fräulein», sagte Brownlow, Rose zuvorkommend; «das Versprechen soll
gehalten werden. Ich glaube nicht, daß es unseren Schritten auch
nur im mindesten hinderlich sein wird. Doch bevor wir bestimmte
Entschließungen in betreff der zu ergreifenden Maßregeln fassen können,
müssen wir notwendig das Mädchen sehen, um von ihr zu hören, ob sie
uns so oder anders dazu verhelfen will oder kann, Monks' habhaft zu
werden, oder wenn nicht, sie wenigstens zu bewegen, uns seine Person zu
beschreiben und uns zu sagen, wo er sich zu verstecken pflegt, oder
wo er sonst zu finden sein mag. Das kann nun vor dem nächsten Sonntag
abend nicht geschehen, und heute ist Dienstag. Mein Rat ist daher, daß
wir uns bis dahin vollkommen ruhig und die Sache selbst vor Oliver
geheimhalten.»

Obgleich Mr. Losberne zu dem Vorschlage, fünf ganze Tage untätig
zu sein, die sauerste Miene machte, so mußte er doch zugeben, im
Augenblick keinen besseren Rat zu wissen; und da sowohl Rose wie Mrs.
Maylie auf Brownlows Seite traten, so wurde des letzteren Rat endlich
allerseits gebilligt.

«Ich nähme gern den Beistand meines Freundes Grimwig in Anspruch»,
sagte Brownlow. «Er ist ein wunderlicher Kauz, besitzt aber sehr
viel Scharfblick und könnte uns von wesentlichem Nutzen sein. Er ist
Rechtsgelehrter von Haus aus und entsagte lediglich aus Unmut darüber,
daß ihm binnen zehn Jahren nur ein einziger Prozeß anvertraut wurde,
dem Advokatenstande. Sie mögen indes selbst entscheiden, ob das eine
Empfehlung ist oder nicht.»

«Ich habe nichts dawider, daß Sie Ihren Freund zuziehen, wenn ich auch
den meinigen zuziehen darf,» sagte Losberne und erwiderte auf Brownlows
Frage, wer derselbe wäre: «Der Sohn Mrs. Maylies und Miß Roses -- sehr
alter Freund.»

Roses Wangen wurden purpurn; sie machte jedoch keine hörbare Einwendung
gegen den Vorschlag (vielleicht weil sie erkannte, daß sie doch
jedenfalls in einer hoffnungslosen Minorität bleiben würde), und Harry
Maylie und Grimwig wurden daher zu Mitgliedern des Komitees ernannt.

«Wir bleiben natürlich so lange in der Stadt,» sagte Mrs. Maylie, «wie
noch die geringste Aussicht vorhanden ist, unsere Nachforschungen
mit Erfolg fortzusetzen. Ich werde bei einer uns alle so sehr
interessierenden Sache weder Mühe noch Kosten sparen und gern
hierbleiben, und wenn es sein muß, zwölf Monate, solange Sie mir sagen
können, daß noch Hoffnung vorhanden ist.»

«Gut», versetzte Brownlow; «und da ich auf Ihren Gesichtern lese, daß
Sie mich fragen wollen, wie es zugegangen ist, daß ich nicht zur Stelle
war, Olivers Erzählung zu bestätigen, und daß ich das Land so plötzlich
verlassen, so erlauben Sie mir, die Forderung zu stellen, daß mir nicht
eher Fragen vorgelegt werden, als bis ich es für geeignet erachte,
denselben durch meine Geschichte zuvorzukommen. Glauben Sie mir, meine
Forderung hat ihren guten Grund; denn wenn ich von ihr abginge, könnte
ich vielleicht Hoffnungen erwecken, welche nie verwirklicht würden und
nur alle schon hinlänglich zahlreichen Schwierigkeiten und Täuschungen
noch vermehrten. -- Meine Herrschaften, wir sind zum Abendessen
gerufen, und Oliver, der einsam im anstoßenden Zimmer weilt, wird am
Ende glauben, daß wir seiner müde geworden wären und einen finsteren
Anschlag ausdächten, ihn wieder in die Welt hinauszustoßen.»

Mit diesen Worten reichte der alte Herr Mrs. Maylie die Hand und führte
sie in das Speisezimmer; Losberne folgte mit Rose, und die Beratung
hatte damit ein Ende.




42. Kapitel.

    Ein alter Bekannter von Oliver läßt entschiedene Geniespuren
    blicken und wird ein öffentlicher Charakter in der Hauptstadt.


Gerade an dem Abende, an welchem Nancy ihre selbstauferlegte Mission
bei Rose Maylie erfüllte, wanderten auf der großen, nach Norden
führenden Heerstraße zwei Personen nach London, denen wir einige
Aufmerksamkeit widmen müssen. Die eine derselben, eine Mannsperson,
gehörte zu den langen, knöchernen Gestalten, die als Knaben wie
verkümmerte Männer, und wenn sie fast Männer sind, wie zu früh groß
gewordene Knaben aussehen. Die zweite, ein Frauenzimmer, war jung, aber
derb und kräftig, was sie auch sein mußte, um unter der schweren Bürde
auf ihrem Rücken nicht zu erliegen. Ihr Begleiter trug nur weniges und
leichtes Gepäck an einem Stocke über der Schulter und konnte daher um
so leichter, zumal da ihm auch die Länge seiner Beine zustatten kam,
stets einige Schritte weit voran sein, woran er es auch ebensowenig
fehlen ließ wie an häufigen Vorwürfen, die er seiner Gefährtin wegen
ihrer Langsamkeit machte. Sie hatten Highgate hinter sich, als er
stillstand und ihr ungeduldig zurief: «Kannst du nicht geschwinder
gehen? Was schleichst du immer so faul von weitem nach, Charlotte?»

«'s ist 'ne schwere Tracht, das kannst du nur glauben», erwiderte sie,
fast atemlos herankommend.

«Schwer? Was ist das für Schwätzen -- wozu hab' ich dich?» fuhr Noah
Claypole (denn er war es) fort und legte sein kleines Bündel auf die
andere Schulter. «Und nun stehst du schon wieder still? Bei dir muß
auch der Beste die Geduld verlieren.»

«Ist es noch weit?» fragte Charlotte, indem ihr die Schweißtropfen über
das Gesicht herabströmten.

«Noch weit? Wir sind schon so gut wie da. Sieh hin -- dort sind die
Lichter von London.»

«Dann sind wir wenigstens noch zwei gute Meilen davon entfernt», sagte
Charlotte verzweiflungsvoll.

«Zwei Meilen oder zwanzig ist auch einerlei; steh auf und mach fort,
oder du bekommst Fußtritte», warnte Noah mit vor Zorn noch mehr als
gewöhnlich geröteter Nase, und Charlotte stand auf und schritt wieder
neben ihm her.

«Wo denkst du für diese Nacht einzukehren, Noah?» fragte sie nach
einiger Zeit.

«Was weiß ich's?» antwortete Mr. Claypole, den das lange Gehen
verdrießlich gemacht hatte.

«Doch in der Nähe?»

«Nein, nicht in der Nähe.»

«Warum denn nicht?»

«Wenn ich dir sage, daß ich das will oder nicht will, so ist's genug,
ohne daß du zu fragen brauchst, warum oder weshalb», entgegnete Noah
mit Würde.

«Ich frage ja nur -- brauchst ja nicht so böse darüber zu werden.»

«Das wär' mir wohl ein recht kluger Streich, im ersten besten
Wirtshause vor der Stadt einzukehren, daß Sowerberry, wenn er uns
etwa nachsetzte, seine alte Nase hereinsteckte und uns gleich wieder
fest hätte und mit Handschellen zurückbrächte! Nein, ich werde in
die engsten Straßen einlenken, die ich finden kann, und nicht eher
haltmachen, als bis wir das entlegenste Gasthaus gefunden haben. Du
kannst deinem Schöpfer danken, daß ich Pfiffigkeit für dich mit habe;
denn wenn wir nicht auf meinen Rat erst den entgegengesetzten Weg
eingeschlagen hätten, so wärst du schon vor acht Tagen eingesperrt, und
dir wäre recht geschehen als 'ner dummen Gans.»

«Ich weiß es, daß ich nicht so klug bin wie du; aber wirf nur nicht
alle Schuld auf mich allein. Wär' ich eingesperrt, würdest du es doch
auch sein.»

«Du weißt doch wohl, daß du das Geld aus dem Ladentische nahmst?»

«O ja, lieber Noah, aber ich nahm es für dich.»

«Nahm ich's hin und trug's bei mir?»

«Nein; du vertrautest mir und ließest 's mich tragen, und das war doch
gut von dir», sagte Charlotte, ihn unter das Kinn klopfend und ihren
Arm in den seinigen legend.

Es verhielt sich in der Tat so; allein es war Mr. Claypoles Weise
nicht, in irgend jemand ein blindes und törichtes Vertrauen zu setzen,
und wir lassen ihm nur Gerechtigkeit widerfahren, wenn wir bemerken,
daß er Charlotten lediglich deshalb so sehr vertraut hatte, damit das
Geld, wenn sie verfolgt würden, bei ihr gefunden werden möchte. Er ließ
sich jedoch bei dieser Gelegenheit natürlich auf keine Darlegung seiner
Beweggründe ein, und beide wanderten im zärtlichsten Einvernehmen
miteinander weiter.

Seinem vorsichtigen Plane zufolge schritt Mr. Claypole, ohne
anzuhalten, bis nach dem Engel von Islington weiter, wo er aus dem
beginnenden Gedränge der Fußgänger und Fuhrwerke sehr scharfsinnig
schloß, daß London nunmehr ernstlich anfinge. Er schaute nun einen
Augenblick umher, welche Straßen die belebtesten und also am meisten zu
meidenden schienen, lenkte in St. Johns Road ein und befand sich bald
tief in dem Gewirr obskurer und schmutziger Straßen und Gassen zwischen
Grays Inn Lane und Smithfield, einem Stadtteile mitten in London, der
trotz der allgemeinen Fortschritte und ungemeiner Verschönerungen
abscheulich geblieben ist.

Noah schaute fortwährend nach einem Gasthause aus, wie er es sich bei
seinen Zwecken und seiner Lage wünschenswert dachte, stand endlich vor
dem elendesten still, das er bis dahin gesehen hatte, und erklärte,
hier für die Nacht einkehren zu wollen.

«Gib mir nun das Bündel,» sagte er, es seiner Begleiterin abnehmend,
«und sprich nicht, außer wenn du angeredet wirst. Wie nennt sich das
Haus? Was steht da -- d-r-e-i --?»

«Krüppel», fiel Charlotte ein.

«Drei Krüppel -- ein sehr guter Name,» bemerkte Noah. «Halt dich dicht
hinter mir -- vorwärts!»

Er stieß die gebrechliche Tür mit den Schultern auf, und beide gingen
hinein. Im Schenkstübchen war niemand, als ein junger Mensch, ein Jude,
der in einem schmutzigen Zeitungsblatte las. Er starrte Noah und Noah
starrte ihn an.

«Sind dies die drei Krüppel?» fragte Noah.

«So nennt sich das Haus.»

«Wir trafen 'nen Gentleman, der uns hierher rekommandiert hat», fuhr
Noah, Charlotte anstoßend, fort, vielleicht, um sie aufmerksam auf
seine List zu machen, sich Achtung zu verschaffen, oder vielleicht um
sie zu erinnern, ihn nicht zu verraten. «Wir möchten hier übernachten.»

«Ich weiß nicht, ob es geht an,» erwiderte Barney -- denn er war der
dienende Geist dieses Hauses -- «will aber anfragen.»

«Bringt uns unterdes in die Gaststube und gebt uns 'nen Mund voll
kaltes Fleisch und 'nen Schluck Bier», sagte Noah.

Barney führte die müden Reisenden in ein Hinterzimmer, brachte ihnen
die geforderten Erfrischungen, teilte ihnen zugleich mit, daß sie über
Nacht bleiben könnten, und ließ das liebenswürdige Pärchen allein. --
Das Zimmer, in welches er sie geführt hatte, befand sich unmittelbar
hinter dem Schenkstübchen und lag einige Fuß niedriger, so daß man aus
jenem, wenn man von einem Diminutivfensterchen etwas hoch in der Wand
einen Vorhang zurückschob, ohne bemerkt zu werden, genau sehen und
hören konnte, was die Gäste darin vornahmen oder sprachen. Noah und
Charlotte hatten sich kaum zu ihrem Imbisse niedergesetzt, als Fagin im
Schenkstübchen erschien, um nach einem seiner jungen Zöglinge zu fragen.

«Pst!» sagte Barney; «es sind nebenan Fremde.»

«Fremde?» wiederholte Fagin flüsternd.

«Ja -- nicht aus der Stadt, kurioses Volk; und ich müßte irren sehr,
wenn sie nicht was wären für Euch.»

Fagin stieg sogleich auf einen Stuhl und sah durch das kleine Fenster,
wie Noah tapfer schmauste und Charlotte von Zeit zu Zeit homöopathische
Dosen zuteilte.

«Aha!» flüsterte Fagin, zu Barney sich umdrehend, «die Miene des
Burschen könnte gefallen mir. Er würde uns sein können nützlich, denn
er versteht's schon, zu kirren die Dirne. Sei stiller als eine Maus,
mein Lieber, daß ich sie höre sprechen.»

Er schaute abermals durch das kleine Fenster, und zwar mit einem
Gesichte, das einem alten Gespenst angehört haben könnte.

«Ich denke also von jetzt ab ein Gentleman zu sein,» sagte Noah,
die Beine ausstreckend und ein Gespräch fortsetzend, dessen Anfang
dem Juden entgangen war. «Nichts mehr von Särgen und Aufwarten bei
Herrschaften, Charlotte, sondern nunmehr wie ein Gentleman gelebt; und
wenn du willst, sollst du 'ne Dame werden.»

«Ei, das möcht' ich freilich wohl, lieber Noah», antwortete Charlotte;
«aber es gibt nicht alle Tage Ladenkassen zu leeren und so, daß man
nachher gut davonkommt.»

«Hol' der Geier alle Ladenkassen!» rief Noah aus, «es gibt noch mehr
Dinge, die geleert werden können.»

«Was meinst du denn?» fragte Charlotte.

«Taschen, Strickbeutel, Häuser, Postkutschen, Banken», erwiderte Mr.
Claypole, dem der Mut wuchs, indem ihm der Porter zu Kopfe stieg.

«Du kannst das aber nicht alles, lieber Noah», sagte Charlotte.

«Ich werde mich nach Genossen umsehen, die es vermögen», versetzte
Noah. «Sie werden uns auf die eine oder andere Weise gebrauchen können.
Du bist selbst soviel wie fünfzig Weibsbilder wert; denn ich hab' nie
eins gekannt, das so voll List und Trug steckte wie du, wenn ich dir
freie Hand lasse.»

«Jemine, wie du flattieren kannst!» rief Charlotte aus und drückte ihm
einen Kuß auf den häßlichen Mund.

«Laß gut sein», sagte Noah mit großer Würde, sich von ihr losmachend;
«sei ja nicht zu zärtlich, wenn ich böse mit dir bin. Ich wollte, daß
ich Hauptmann 'ner Bande wär', hätte sie unter der Zucht und folgte
ihnen allerwärts nach, ohn' daß sie's selber wüßten. Das wär' so
was für mich, wenn's guten Profit abwürfe; und hör', wenn's uns nur
glückte, daß uns einige Gentlemen von dieser Sorte in den Wurf kämen,
es wär' uns soviel wert wie unsere Zwanzigpfundnote -- besonders da wir
eigentlich nicht wissen, wie wir sie loswerden sollen.»

Mr. Claypole blickte bei diesen Worten mit äußerst weiser Miene in den
Porterkrug hinein, trank, nickte Charlotte herablassend zu und stand
im Begriff, einen zweiten Zug zu tun, als die Tür sich auftat und die
Erscheinung eines Unbekannten ihn unterbrach. Der Unbekannte war Mr.
Fagin. Er hatte seine einnehmendste Miene angenommen, näherte sich mit
einer sehr tiefen Verbeugung, nahm an einem Tische dicht neben dem, an
welchem das Pärchen saß, Platz und rief dem grinsenden Barney zu, ihm
einen Trunk zu bringen.

«Ein angenehmer Abend, Sir, nur kühl für die Jahreszeit», hub er
händereibend an. «Sie kommen vom Lande herein, wie ich sehe, Sir?»

«Woran sehen Sie denn das?» fragte Noah Claypole.

«Wir haben nicht in London soviel Staub, wie Sie mitbringen», erwiderte
der Jude, nach Noahs und Charlottes Schuhen und den Bündeln hinzeigend.

«Sie sind mir ein pfiffiger Gesell», versetzte Noah mit Lachen. «Hör'
nur an, was er sagt, Charlotte.»

«Ei nun, mein Lieber, man muß wohl sein pfiffig in dieser Stadt»,
fuhr der Jude, vertraulich flüsternd, mit dem Finger an die Nase
schlagend, fort, -- eine Geste, die Noah sogleich nachahmte, doch
nicht mit vollständigem Gelingen, da seine Nase nicht groß genug dazu
war. Fagin schien jedoch den Versuch so auszulegen, als wenn ihm Noah
vollkommen hätte beipflichten wollen, und schob dem letzteren sehr
freundschaftlich den soeben von Barney kredenzten Krug zu.

«Gutes Getränk», entgegnete Fagin. «Wer es will immer trinken, muß
immer leeren etwas, eine Ladenkasse, eine Tasche, einen Strickbeutel,
ein Haus, eine Postkutsche oder eine Bank.»

Mr. Claypole sank rückwärts auf seinen Stuhl und wandte sein kreideweiß
gewordenes und grenzenlos bestürztes Gesicht vom Juden nach Charlotten.

«Seien Sie ohne Sorgen meinetwegen, mein Lieber», sagte Fagin, näher
rückend. «Ha, ha, ha! -- es war ein Glück, daß niemand Sie hörte, als
ich zufällig -- es war ein großes Glück für Sie.»

«Ich nahm's nicht heraus», stotterte Noah, die Füße nicht mehr wie ein
unabhängiger Gentleman ausstreckend, sondern so tief unter den Stuhl
ziehend, als er konnte. «Sie hat's ganz allein getan, und du hast's
Charlotte; du weißt, daß du's hast.»

«Es ist gleichviel, mein Lieber, wer es hat oder wer es tat», fiel der
Jude ein, doch nichtsdestoweniger mit Falkenaugen nach dem Mädchen und
den beiden Bündeln hinblickend. «'s ist mein Geschäft auch, und Sie
gefallen deswegen mir.»

«Was ist Ihr Geschäft?» fragte Noah, sich einigermaßen wieder fassend.

«Nun, dasselbe, das angefangen haben Sie,» antwortete Fagin, «und die
Wirtsleute hier treiben es auch. Sie sind eingegangen zur rechten Tür
und sind hier so sicher wie in Abrahams Schoß. Es gibt kein sichereres
Haus in der Stadt als die Krüppel; das heißt, wenn ich's will, und ich
habe gefaßt eine Neigung zu Ihnen und dem jungen Frauenzimmer. Sie
wissen nun Bescheid und können sich beruhigen vollkommen.»

Noah blickte ihn trotz dieser Versicherung noch immer furchtsam und
argwöhnisch an und rückte unruhig auf seinem Stuhle hin und her. Fagin
nickte Charlotte freundlich zu, sprach ihr leise Mut zu und fuhr fort:
«Ich will Ihnen sagen noch mehr. Ich hab' einen Freund, der Ihren
Herzenswunsch, glaub' ich, kann befriedigen und Ihnen Gelegenheit
geben, zu arbeiten vorerst in dem Geschäftszweige, der Ihnen gefällt am
besten, und Sie lehren alle anderen.»

«Sie sprechen, als wenn es Ihr Ernst wäre,» bemerkte Noah.

«Wenn ich nicht spräche im Ernst, welchen Nutzen könnt' ich haben
davon?» versetzte der Jude achselzuckend. «Kommen Sie -- lassen Sie
mich reden mit Ihnen ein Wörtchen draußen!»

«Es tut nicht not, daß wir uns die Mühe geben, hinauszugehen», sagte
Noah, die Beine allmählich wieder unter dem Stuhle hervorziehend. «Sie
kann unterdes das Reisegepäck in unsere Kammer tragen. Charlotte,
bring' die Bündel hinauf!»

Charlotte gehorchte dem mit großer Würde gegebenen Befehle ohne die
mindeste Zögerung, hob die beiden Bündel auf und ging hinaus.

«Hab' ich sie nicht ganz gut in der Zucht?» fragte Noah im Tone eines
Wärters, der ein wildes Tier gezähmt hat.

«Oh, vortrefflich», erwiderte Fagin, ihn auf die Schulter schlagend.
«Sie sind ein Genie, mein Lieber.»

«Würde schwerlich hier sein, wenn ich's nicht wäre», versetzte Noah.
«Doch verlieren Sie keine Zeit, denn sie wird bald wieder da sein.»

«Sehr wohl! Was meinen Sie -- wenn Ihnen gefiele mein Freund, was
könnten Sie tun Besseres, als zu treten mit ihm in Verbindung?» sagte
Fagin.

«Es kommt darauf an, ob er gute Geschäfte macht», entgegnete Noah, dem
Juden mit dem einen seiner kleinen Augen pfiffig zublinzelnd.

«Er beschäftigt eine Menge Leute und hat die beste Gesellschaft von
allen, die treiben das Geschäft.»

«Echte Stadtbursche?»

«'s ist kein Nicht-Lond'ner drunter, und er würde Sie nicht einmal
annehmen, selbst auf meine Empfehlung nicht, wenn es ihm nicht fehlte
eben jetzt an Gehilfen.»

«Würd' ich 'rausrücken müssen?» fragte Noah, an seine
Beinkleidertaschen schlagend.

«Ohne zwanzig Pfund ging's an unmöglich», erwiderte Fagin auf das
bestimmteste.

«Aber zwanzig Pfund -- 's ist ein Haufen Geld!»

«Eine Kleinigkeit, wenn Sie nicht können los werden die Banknote.»

«Wann könnt' ich Ihren Freund sehen?»

«Morgen früh.»

«Wo?»

«Hier.»

«Hm! -- Wie hoch ist der Lohn?»

«Sie leben wie ein Gentleman -- haben Kost und Wohnung und Tabak und
Branntwein frei -- die Hälfte von allem, was Sie verdienen und was das
junge Frauenzimmer verdient.»

Es ist sehr zweifelhaft, ob Noah Claypole, so sehr bedeutend seine
Habgier auch war, auf diese glänzenden Bedingungen eingegangen sein
würde, wenn er hätte vollkommen frei handeln können; allein er
bedachte, daß es, wenn er nein sagte, in der Gewalt seines neuen
Bekannten stand, ihn augenblicklich den Händen der Gerechtigkeit zu
überliefern. Er erklärte daher, daß ihm der Vorschlag Fagins nicht ganz
unannehmbar erschiene.

«Aber sehen Sie,» setzte er hinzu, «da sie imstande ist, ein gutes
Stück Arbeit auszurichten, so möcht' ich etwas recht Leichtes zugeteilt
bekommen. Was würde jetzt wohl für mich passen? Es müßte so etwas sein,
wobei ich mich nicht eben anzustrengen brauche und wobei keine Gefahr
wäre.»

«Mein Freund braucht jemand, der was Rechtes leisten könnte im
Spionierfache -- was sagen Sie dazu?» entgegnete der Jude.

«Gefällt mir nicht ganz übel, und bisweilen möcht' ich wohl darin
arbeiten», sagte Noah zögernd; «aber es wirft nur für sich selber
nichts ab, wissen Sie.»

«Freilich», pflichtete Fagin ihm bei. «Was sagen Sie zu den alten
Damen? Ihnen nehmen die Strickbeutel und Pakete und dann laufen um die
Ecke -- damit wird gemacht viel Geld.»

«Schreien diese aber nicht fürchterlich, oder kratzen auch bisweilen?»
wandte Noah kopfschüttelnd ein. «Ich habe keine Lust dazu. Ist kein
anderes Fach offen?»

«Halt, ja!» sagte der Jude, die Hand auf Noahs Knie legend. «Das
Schratzchenbehandeln!»

«Was ist denn das?»

«Die Schratzchen sind die kleinen Kinder, die mit Sixpencen und
Schillingen ausgeschickt werden von ihren Müttern, um einzuholen
allerhand; und das Behandeln ist wegnehmen ihnen das Geld, das sie
immer haben in den Händen, und sie dann stoßen in die Straßenrinne und
ganz langsam davongehen, als wenn geschehen wäre nichts, als daß ein
Kind wäre gefallen und hätte sich Schaden getan ein wenig. Ha, ha, ha!»

«Ha, ha, ha!» stimmte Mr. Claypole ein und warf außer sich vor
Vergnügen die Füße hoch in die Luft. «Beim Deuker, ja, das ist das
Rechte!»

«Gewiß, gewiß», sagte Fagin; «Sie können haben prachtvolle Bezirke
in Camden-Town und Battle-Bridge und solchen Gegenden mehr, wo immer
ausgeschickt werden viele und zu jeder Tagesstunde niederwerfen so
manche Schrätzchen, wie Sie wollen nur.»

«Ich bin alles wohl zufrieden», sagte Noah, als er sich von seiner
Ekstase wieder erholt hatte und Charlotte zurückgekehrt war. «Welche
Zeit bestimmen wir auf morgen?»

«Nun, belieben Sie zehn Uhr?»

Noah nickte.

«Welchen Namen soll ich nennen meinem Freunde?»

«Mr. Bolter; Mr. Morris Bolter -- dies ist Mrs. Bolter.»

«Ich bin Mrs. Bolters gehorsamer Diener», sagte Fagin, sich mit
grotesker Galanterie verbeugend. «Ich hoffe, Sie recht bald noch besser
kennen zu lernen.»

«Hörst du, was der Herr sagt, Charlotte?» herrschte ihr Mr. Claypole zu.

«Ja, lieber Noah», antwortete Charlotte, die Hand ausstreckend.

Mr. Morris Bolter, sonst Claypole, wandte sich zu dem Juden und sagte:
«Noah ist der Schmeichelname, den sie mir gibt.»

«Oh, ich verstehe -- verstehe vollkommen», erwiderte Fagin, für diesmal
die Wahrheit redend. «Gute Nacht! Gute Nacht!»




43. Kapitel.

    In welchem berichtet wird, wie sich der gepfefferte Baldowerer in
    Verlegenheiten benahm.


«Also Ihr selber waret Euer Freund -- nicht wahr?» fragte Mr. Bolter,
sonst Claypole, als er, nach zwischen ihm und Fagin besiegeltem
Vertrage, in des Juden Wohnung geführt worden war. «Dummkopf, der ich
bin -- ich hätt's mir doch gestern abend schon denken können!»

«Jedermann ist sein eigener Freund», erwiderte Fagin. «Es gibt
Tausendkünstler, die da sagen, Nummer Drei wäre die Zauberzahl, und
andere sagen Nummer Sieben. Aber es ist nicht wahr, mein Freund. Nummer
Eins ist's!»

«Ha, ha, ha! Nummer Eins für immer!»

«In einer kleinen Genossenschaft, wie die unsrige ist,» sagte der Jude,
der eine Erklärung für nötig hielt, «haben wir eine allgemeine Nummer
Eins; das will sagen, Ihr könnt Euch nicht betrachten als Nummer Eins,
ohne mich und all die anderen jungen Leute als dieselbe zu betrachten
zugleich.»

«Das wär' der Teufel!»

«Ihr seht wohl,» fuhr der Jude fort, sich anstellend, als ob er die
Unterbrechung nicht beachtete, «unser Nutzen und Schaden ist eins so
ganz, daß es nicht sein kann anders. Zum Beispiel, es ist Euer Zweck
und Absicht, zu sorgen für Nummer Eins -- das heißt für Euch selbst.»

«Ganz recht, ganz recht.»

«Sehr wohl -- Ihr könnt aber nicht sorgen für Euch selber, Nummer Eins,
ohne zugleich zu sorgen für mich, Nummer Eins.»

«Nummer Zwei wollt Ihr sagen», fiel Mr. Bolter ein, der die Tugend der
Selbstliebe im allerhöchsten Maße besaß.

«Nein, nein!» entgegnete der Jude. «Ich bin von derselben Wichtigkeit
für Euch, wie Ihr es seid selbst.»

«Hört,» unterbrach Mr. Bolter, «Ihr seid ein sehr netter Mann, und ich
halte sehr viel von Euch; aber so dicke Freunde, wie Ihr mit dem allen
meint, sind wir doch noch nicht.»

«Bedenkt doch, bedenkt doch nur!» sagte der Jude achselzuckend und die
Hände ausstreckend. «Ihr habt getan, was sehr hübsch war, und ich ehre
und liebe Euch deshalb; aber 's ist auch derart, daß es Euch zugleich
einbringen kann die Krawatte, die so leicht ist einzuknüpfen und so
schwer wieder aufzubinden -- den Strick nämlich!»

Mr. Bolter legte die Hand an sein Halstuch, als wenn es ihm unbequem
eng säße, und murmelte eine Art von Beistimmung.

«Der Galgen,» fuhr Fagin fort, «der Galgen, mein Lieber, ist ein
häßlicher Wegweiser, der zeigt um eine sehr scharfe Ecke und hat
gemacht ein Ende der Weiterreise vieler mutvoller, wackerer Leute auf
der großen Heerstraße. Euch zu halten auf der bequemen Straße und zu
bleiben dem Galgen fern, muß sein Euer Nummer Eins, mein Lieber.»

«Natürlich», fiel Mr. Bolter ein; «aber wozu redet Ihr von solchen
Dingen?»

«Bloß um Euch zu zeigen meine Meinung deutlich», erwiderte Fagin, die
Augenbrauen emporziehend. «Ihr könnt das nicht allein, sondern hängt
dabei ab von mir, und ich hänge ab von Euch, wenn mein kleines Geschäft
soll haben guten Fortgang. Das erste ist Eure Nummer Eins, das zweite
ist meine Nummer Eins. Je mehr Euch liegt am Herzen Eure Nummer Eins,
desto mehr müßt Ihr sein besorgt für meine; und so kommen wir endlich
wieder zurück auf das, was ich Euch sagte gleich anfangs -- daß Sorge
für Nummer Eins kommt uns allen zugut, und lassen wir's fehlen daran,
gehen wir zugrunde miteinander alle.»

«Das ist wohl wahr», bemerkte Bolter gedankenvoll. «Ihr seid, meiner
Treu, ein geriebener alter Gesell!»

Fagin erkannte mit innigstem Vergnügen, daß dies keine bloße
Schmeichelei war, sondern daß er seinem Rekruten eine bedeutende
Vorstellung von seiner Verschlagenheit und Gewalt beigebracht hatte,
was beim Beginn ihrer beiderseitigen Bekanntschaft von großer
Wichtigkeit war. Um den Eindruck, den er auf den jungen Menschen
gemacht hatte, noch zu verstärken, ließ er ihn einige Blicke in
die Großartigkeit und den Umfang seiner Operationen tun, wobei er,
seinem Zwecke gemäß, Wahrheit und Dichtung so geschickt miteinander
vermischte, daß Mr. Bolters Hochachtung gegen ihn sichtlich zunahm und
er zugleich eine Zutat heilsamer Furcht erhielt, welche bei ihm zu
erwecken äußerst wünschenswert war.

«Dies gegenseitige Vertrauen ist es,» sagte der Jude, «was mich tröstet
wegen schwerer Verluste. Erst gestern morgen verlor ich meinen besten
Gehilfen.»

«Ist er Euch davongegangen?» fragte Mr. Bolter.

«Ganz wider seinen Willen», antwortete Fagin. «Er war beschuldigt des
Versuchs eines Taschendiebstahls, und sie fanden bei ihm eine silberne
Schnupftabaksdose. Es war seine eigene, mein Lieber, seine eigene, denn
er schnupfte selbst, und die Dose war ihm sehr wert. Er ward wieder
vorbeschieden auf heute, denn sie meinten herbeischaffen zu können den
Eigentümer. Oh, er war wert fünfzig silberne Dosen, und ich würde sie
darum geben, wenn ich ihn hätte wieder. Ihr solltet gekannt haben den
Baldowerer, mein Lieber; solltet gekannt haben den Gepfefferten!»

«Ich hoffe ihn noch kennen zu lernen -- meint Ihr nicht, Fagin?»

«Ich muß es bezweifeln», erwiderte der Jude seufzend. «Wenn vorgebracht
wird kein neues Zeugnis gegen ihn, so werden wir ihn freilich haben
wieder nach ein sechs oder acht Wochen; sonst aber wird er gerumpelt,
und auf lebenslang, sicher auf lebenslang; denn sie wissen's, welch ein
gescheiter Bursch ist der Baldowerer.»

«Was wollt Ihr damit sagen, daß er gerumpelt würde?» fragte Mr. Bolter.
«Warum sprecht Ihr in solchen Ausdrücken zu mir, da Ihr doch wißt, daß
ich sie nicht verstehen kann?»

Fagin war im Begriff, ihm zu sagen, daß Rumpeln soviel als Deportieren
bedeute, allein in demselben Augenblicke trat Master Bates mit den
Händen in den Beinkleidertaschen und einem halbkomisch-trübseligen
Gesichte herein.

«'s ist vorbei mit ihm, Fagin», sagte er, nachdem er seinem neuen
Kameraden gebührend vorgestellt worden war.

«Was willst du sagen damit?» fragte der Jude mit bebenden Lippen.

«Sie haben den Herrn ausgespürt, dem die Dose gehörte, und noch mehrere
Anklagen vorgebracht -- der Gepfefferte erhält freie Überfahrt»,
antwortete Master Bates. «Ich muß ä vollständ'gen Traueranzug haben,
Fagin, und ä Hutband, ihn zu besuchen, eh' er seine Reise antritt.
's is doch die Möglichkeit! -- Jack Dawkins -- der große Jack Dawkins
-- der Baldowerer -- der gepfefferte Baldowerer -- und wird gerumpelt
wegen 'ner lumpigen Schnupftabaksdose! -- wegen 'ner erbärmlichen
Dorfdruckerei[AS]. Ich hätt's nimmermehr geglaubt, daß er's unter 'ner
goldenen Uhr mit Kette und Petschaften zum mind'sten getan haben würde.
Nein, wenn er noch 'nem reichen alten Herrn seine ganze Massumme[AT]
und alles geganft[AU] hätte, so daß er doch abreiste wie ä Gentleman!
-- aber so -- wie ä gemeiner Dorfdrucker! -- ohne Ruhm, ohne Ehre!»

  [AS] Taschendiebstahl.

  [AT] Geld.

  [AU] geraubt.

Also seine Gefühle für den unglücklichen Freund ausdrückend, nahm
Master Bates entrüstet und niedergeschlagen auf dem ersten besten
Stuhle Platz.

«Was schwätzest du, daß er hätte weder Ruhm noch Ehre!» rief Fagin,
seinem Zöglinge einen zornigen Blick zuwerfend, aus. «Ist er nicht
immer gegangen über euch allen -- hat's einer von euch ihm tun können
gleich -- nur von fern tun können gleich -- wie?»

«Freilich, freilich! Aber sollt's einen denn nicht jammern,» entgegnete
Charley, «sollte man nicht des Teufels werden vor Verdruß, daß nichts
davon vor Gericht verlautet, daß niemand nur zur Hälfte erfährt,
wer und was er gewesen ist? Welch 'nen elenden Titel wird er im
Newgatekalender bekommen -- kommt vielleicht nicht mal 'nein! O weh, o
weh, was es für ä Jammer ist!»

«Ah! wenn du's so meinst,» sagte der Jude mit vergnügtem Kichern und
ihm die Hand reichend, «wenn du's so meinst, das ist ein andres.
Schaut, mein Lieber,» fuhr er, zu Bolter sich wendend, fort, «schaut,
wie stolz sie sind auf ihren Stand und Beruf! Ist es nicht zu sehen
eine Lust?»

Mr. Bolter nickte Beistimmung, und der Jude trat mit freudigem Stolze
zu Charley, klopfte ihm auf die Schulter und sagte tröstend: «Sei nur
ohne Sorgen, Charley; es wird schon kommen an den Tag, und er wird's
selbst schon zeigen, was er ist gewesen, und wird keine Schande bringen
über seine alten Kameraden und Lehrer. Bedenkt auch nur, wie jung er
noch ist! Ist's nicht schon eine große Auszeichnung, bei seinen Jahren
gerumpelt zu werden auf lebenslang?»

«Ja freilich -- daran hatt' ich nicht gedacht -- 's ist freilich
ehrenvoll genug!» erwiderte Charley, einigermaßen getröstet.

«Er wird haben alles, was er braucht», fuhr der Jude fort; «wird in
Doves[AV] gehalten werden wie ein Gentleman, alle Tage haben sein Bier
und Geld in der Tasche, zu spielen Bild oder Schrift, wenn er's nicht
kann ausgeben.»

  [AV] Gefängnis.

«Wahr und wahrhaftig?» rief Charley aus.

«Ganz gewiß, ganz gewiß!» sagte Fagin. «Und wir werden ihm schaffen
'nen Advokaten -- den zungenfertigsten, der wird sein zu finden -- zu
führen seine Verteid'gung; und wenn er will, kann er auch halten eine
Rede selbst, und wir werden's lesen alles in den Blättern. Was sagst
du, Charley?»

«Prächtig, prächtig!» rief Master Bates aus. «Oh, es ist mir, als
wenn ich ihn vor mir sähe, wie er die alten Perücken bei der Nase
herumzieht, wie sie sich abstrap'zieren, wichtig und feierlich
aussehen, und er so vertraulich und gemütlich zu ihnen spricht, als
wenn er des Richters eigener Sohn wäre und 'ne Rede bei Tisch hielte --
ha, ha, ha!»

«Aber Charley,» sagte der Jude, «wir müssen ersinnen, ein Mittel in
Erfahrung zu bringen, wie er sich macht heute und was ihm passiert.»

«Soll ich hingehen?» entgegnete Charley eifrig, denn er versprach sich
jetzt den köstlichsten Genuß von einem Schauspiele, bei welchem der
Baldowerer, den er noch vor kurzem als einen Gegenstand des Mitleides
und Verdrusses betrachtet, in der ersten glänzenden Rolle auftreten
sollte.

«Nicht um alles in der Welt», antwortete Fagin.

«So schickt den da -- den Neuangeworbenen hin», riet Charley; «den
kennt niemand.»

«Kein schlechter Rat», sagte der Jude. «Was meint Ihr, mein Lieber?»

«Nein, nein», erwiderte Mr. Bolter kopfschüttelnd; «nichts davon, 's
ist mein Fach nicht.»

«Was habt Ihr ihm denn für ä Fach zugeteilt, Fagin?» fragte Charley
Bates, Noahs schlottrige Gestalt mit großem Widerwillen betrachtend.
«Sich den Rücken zu decken, wenn was zu riskieren ist, und alles
aufzuessen, wenn wir in guter Ruhe zu Haus sitzen?»

«Geht dich nichts an», fiel Mr. Bolter ein; «und nimm dir keine
Freiheiten heraus gegen Leute, die über dir sind, Knirps, oder du wirst
erfahren, daß du vor die unrechte Schmiede gekommen bist.»

Master Bates belachte die prahlerische Drohung so ausgelassen, daß es
einige Zeit währte, bevor Fagin vermitteln und Mr. Bolter vorstellen
konnte, daß er bei einem Besuche des Polizeiamts durchaus keine Gefahr
liefe; denn von seiner kleinen Affäre würde noch ebensowenig Kunde nach
der Hauptstadt, wo man ihn am wenigsten vermute, gelangt sein, wie
ein Steckbrief; und sollte es das Unglück gewollt haben, so würde er
sich, gut verkleidet, nirgends in ganz London mit größerer Sicherheit
aufhalten können, als eben auf der Polizei, wo er ohne Zweifel am
letzten gesucht werden dürfte.

Mr. Bolter ließ sich endlich durch diese und ähnliche Vorstellungen,
noch mehr aber durch seine Furcht vor dem Juden bewegen, freilich mit
der verdrießlichsten Miene, einzuwilligen, die Sendung zu übernehmen.
Fagin versah ihn sogleich mit einem Kärrnerkittel, manchesternen
Kniehosen, ledernen Beinlingen, einem Hut mit Weggeldzetteln und
einer Peitsche, und zweifelte um so weniger am Erfolge, da Mr. Bolter
obendrein die Ungelenkheit eines Kärrners im vollkommensten Maße besaß.
Der Baldowerer wurde ihm genau beschrieben, und Master Bates geleitete
ihn durch Nebengassen nach Bow-Street, wies ihn zurecht, erteilte ihm
jede sonst nötige Auskunft, forderte ihn zur Eile auf und versprach,
seine Rückkehr an der Stelle, wo er ihn verließ, zu erwarten.

Master Bates' Weisungen waren so genau gewesen, daß sich Noah Claypole
oder Morris Bolter, wie der Leser will, sehr leicht, und ohne fragen
zu müssen, zurechtfand. Er drängte sich durch einen hauptsächlich
aus Frauenzimmern bestehenden Haufen hinein in das düstere und
schmutzige Gerichtszimmer. Vor den Schranken standen ein paar Weiber,
die ihren bewundernden Angehörigen oder Bekannten zunickten, während
der Gerichtsschreiber zwei Polizisten und einem einfach gekleideten,
über den Tisch lehnenden Manne Zeugenaussagen vorlas und ein
Gefängniswärter lässig dastand und von Zeit zu Zeit Ruhe oder «das Kind
hinauszuschaffen» gebot, wenn ein ungebührliches Geflüster oder der
Aufschrei eines Säuglings eine Störung verursachte. Noah blickte scharf
umher nach dem Baldowerer, bemerkte Leute genug, welche Geschwister
oder Eltern des Gepfefferten hätten sein können, aber niemand, auf den
die Beschreibung gepaßt hätte, die ihm von Jack Dawkins selbst gegeben
worden war. Endlich waren die vor den Schranken stehenden Frauenzimmer
abgeurteilt und entfernt, und nunmehr erschien ein Angeklagter, der
ohne Frage der Baldowerer war.

Jack ging vor dem Gefängniswärter, den Hut in der rechten Hand haltend
und die Linke in der Beinkleidertasche, keck genug einher und fragte,
sobald er auf der Anklagebank stand, sogleich mit hörbarer Stimme,
warum man ihn an die schimpfliche Stelle geführt habe.

«Willst du wohl den Mund halten?» rief ihm der Schließer zu.

«Bin ich kein Engländer?» rief der Baldowerer zurück. «Wo sind meine
Freiheiten?»

«Wirst sie bald genug bekommen», entgegnete der Schließer, «und zwar
mit Pfeffer dazu.»

«Je nun, wenn sie mir gekränkt werden, so wird sich's schon
finden, was der Staatssekretär für die inneren Angelegenheiten den
Oberschenkeln[AW] zu sagen hat», fuhr Jack Dawkins fort. «Jetzo aber --
holla, was gibt's hier? Wollen die Friedensrichter nicht so gut sein,
diese kleine Sache abzumachen und mich nicht aufzuhalten, indem sie die
Zeitungen lesen? Ich hab' 'nen Gentleman nach der City bestellt, bin
ein Mann von Wort und auch sehr pünktlich in Geschäften; er wird daher
fortgehen, wenn ich nicht zur bestimmten Zeit da bin, und es könnte
'ne Klage auf Schadenersatz geben gegen die, die mich aufgehalten
haben. -- He, Binnfaden[AX], wie heißen die beiden Abrosche[AY] da
auf der Richterbank?» wandte er sich zu dem Gefängniswärter, was die
Nächststehenden dermaßen kitzelte, daß sie fast so herzlich lachten,
wie es Master Bates selbst getan haben würde, wenn er die spaßhafte
Frage gehört hätte.

  [AW] Richtern.

  [AX] Amtsdiener.

  [AY] Spitzbuben.

«Ruhe da!» rief der Schließer.

Einer der Friedensrichter fragte nach der Ursache des Geräusches.

«Hier steht ein Taschendieb, Ihr Edlen.»

«Ist der Knabe schon hier gewesen?»

«Hätt's schon manchmal sein sollen, Ihr Edlen. Überall sonst ist er
schon lange genug gewesen. Ich kenne ihn sehr wohl, Ihr Edlen.»

«So! Ihr kennt mich also?» rief der Baldowerer, sich anstellend, als
wenn er die Angabe aufzeichnete. «Sehr wohl. Das setzt 'ne Klage wegen
Beschimpfung meines guten Namens.»

Es wurde abermals gelacht und abermals Ruhe geboten.

«Wo sind die Zeugen?» begann der Gerichtsschreiber.

«Ach, so ist's recht!» fiel Jack Dawkins ein. «Ja, wo sind die Zeugen?
Ich möchte doch das Pläsier haben, sie zu sehen!»

Sein Wunsch wurde augenblicklich erfüllt, denn es trat ein Polizist
vor, der gesehen hatte, daß der Angeklagte einem Herrn das Taschentuch
aus der Tasche gezogen, und da es ein sehr altes gewesen, nachdem er
Gebrauch davon gemacht, wieder hineingesteckt hatte. Er hatte deshalb
den Täter verhaftet und bei demselben eine silberne Schnupftabaksdose
mit dem Namen des Eigentümers auf dem Deckel gefunden. Der Eigentümer
der Dose war gleichfalls gegenwärtig, beschwor, daß die Dose die
seinige wäre, und daß er sie vermißt hätte, sobald er sich Bahn aus
dem Gedränge gemacht, in welchem (wie sich fand) der Angeklagte das
fragliche Taschentuch entwendet und zurückgegeben. Er hatte auch
bemerkt, daß sich ein junger Gentleman eiligst von ihm entfernt, und
der junge Gentleman war eben der Baldowerer.

«Hast du eine Frage an den Zeugen zu richten, Knabe?» fragte der
Friedensrichter.

«Ich mag mich nicht erniedrigen, mit ihm in Unterredung zu treten»,
entgegnete Jack Dawkins.

«Hast du überhaupt was zu sagen?»

«Hörst du die Frage Seiner Edlen nicht, ob du etwas zu sagen hättest?»
fiel der Schließer, den stummen Baldowerer mit dem Ellbogen anstoßend,
ein.

«Bitt' um Vergebung», sagte Jack, zerstreut aufblickend. «Redeten Sie
mich an?»

«Ihr Edlen,» bemerkte der Schließer, «ich hab' mein Lebtag noch keinen
solchen jungen Erzspitzbuben gesehen. Willst du was sagen, Bursch?»

«Nein,» entgegnete der Baldowerer, «hier nicht; dies ist nicht das
rechte Kaufhaus für die Gerechtigkeit, und außerdem frühstückt
mein Advokat heute morgen bei dem Vizepräsidenten des Hauses der
Gemeinen. Jedoch werden wir, ich und er und eine sehr reputierliche
Bekanntschaft, anderwärts sprechen, und zwar so, daß die
Richterperücken wünschen werden, daß sie niemals geboren oder daß sie
von ihren Bedienten aufgehängt sein möchten, statt mich hier heute
morgen zu prozessieren. Ich will --»

«Er ist vollständig überführt; ins Gefängnis mit ihm -- man bringe ihn
hinaus!» rief der Gerichtsschreiber.

«Komm her, Bursch», sagte der Schließer.

«Komme schon», sagte der Baldowerer, seinen Hut mit der flachen Hand
glättend, und wandte sich darauf nach der Richterbank: «Es hilft Ihnen
nichts, Gentlemen, und wenn Sie auch noch so bestürzt aussehen. Ich
werde kein Erbarmen mit Ihnen haben, für keinen Heller nicht. Sie
werden dafür büßen, und ich möchte um vieles nicht an Ihrer Stelle
sein. Ich würde die Freiheit nicht annehmen, und wenn Sie mich auf den
bloßen Knien darum anflehten. Binnfaden, führ mich ab ins Gefängnis!»

Der Schließer zog ihn beim Kragen heraus, Jack drohte, die Sache
vors Parlament zu bringen, und lächelte darauf den Schließer mit der
behaglichsten Selbstzufriedenheit an.

Sobald ihn Noah hatte fortschleppen sehen, eilte er zu Master Bates
zurück, der ihn in einem angemessenen Verstecke erwartet hatte und sich
zeigte, sobald er sich vergewissert, daß niemand seinem neuen Bekannten
nachfolgte. Sie gingen schleunigst miteinander nach Hause, um Fagin
die erfreuliche Kunde zu bringen, daß sich der Baldowerer vollkommen
ehrenhaft benommen und sich einen glänzenden Namen gemacht habe.




44. Kapitel.

    Nancy wird verhindert, ihr Rose Maylie gegebenes Versprechen zu
    erfüllen.


Wie vollkommen eingeweiht Nancy in alle Verstellungskünste auch war,
vermochte sie doch die Gemütsbewegungen nicht gänzlich zu verbergen,
die das Bewußtsein ihres Schrittes bei ihr hervorbrachte. Sie erinnerte
sich, daß sowohl der listige Jude wie der brutale Sikes sie in das
Geheimnis von Anschlägen, die sie vor allen anderen verborgen hielten,
eingeweiht hatten, und zwar im vollkommensten Vertrauen auf ihre Treue
und über allen Verdacht erhabene Zuverlässigkeit; und so schändlich
jene Anschläge, so ruchlos die Urheber derselben sein mochten, so
erbittert sie selbst gegen den Juden war, der sie Schritt für Schritt
tiefer und immer tiefer in einen Abgrund von Verbrechen und Elend
geführt hatte, aus welchem kein Entrinnen möglich war: es gab doch
Augenblicke, wo bei ihr eine mildere Stimmung gegen ihn vorherrschte
und der Gedanke ihr Unruhe verursachte, daß ihn endlich infolge der
von ihr gemachten Enthüllung sein lange vermiedenes, aber freilich
vollkommen verdientes Schicksal ereilen möchte.

Doch waren dies nur vorübergehende Gedanken und Gefühle bei ihr, deren
sie sich aus Macht der Gewohnheit nicht gänzlich zu erwehren imstande
war; denn ihr Entschluß stand fest, und ihr Charakter war derart,
daß sie sich durch keinerlei Rücksichten bewegen ließ, einen einmal
gefaßten Entschluß wieder aufzugeben. Ihre Besorgnis für Sikes würde
ein noch stärkerer Beweggrund gewesen sein, zurückzutreten, solange es
noch Zeit war; allein sie hatte es sich ausbedungen, daß ihr Geheimnis
streng bewahrt werden sollte -- hatte keinen Faden an die Hand gegeben,
der zu seiner Entdeckung führen konnte -- hatte um seinetwillen sogar
das Anerbieten einer Zuflucht vor allem sie umgebenden Laster und
Elend zurückgewiesen -- und was konnte sie mehr tun? Sie war und blieb
entschlossen.

Obgleich aber alle ihre inneren Kämpfe so endeten, erneuerten sie sich
doch fortwährend und ließen auch ihre Spuren zurück. Nach wenigen
Tagen sah sie blaß und abgezehrt aus. Bisweilen beachtete sie gar
nicht, was um sie her vorging, und nahm an Gesprächen keinen Teil, bei
welchen sie sonst die Lebhafteste und Lauteste gewesen sein würde; und
bisweilen lachte sie wieder ohne Heiterkeit und lärmte ohne Zweck und
Veranlassung. Zu anderen Zeiten -- und oft einen Augenblick darauf --
saß sie schweigend, niedergeschlagen, hinbrütend, den Kopf auf die
Hände gestützt, da, während gerade die Anstrengung, womit sie sich dann
wieder aufraffte, noch stärker verkündete, daß sie Unruhe empfand und
daß ihre Gedanken mit ganz anderen Dingen als denen beschäftigt waren,
die von ihren Gesellschaftern besprochen wurden.

Der Sonntagabend war gekommen, und die Glocke der nächsten Kirche
schlug elf. Sikes und der Jude unterbrachen ihr Gespräch und horchten
-- und aufblickend und noch gespannter horchte Nancy.

«'ne Stunde vor Mitternacht», sagte Sikes, das Fenster öffnend und nach
seinem Stuhle zurückkehrend; «auch ist's neblig und finster -- 'ne gute
Geschäftsnacht.»

«Ah, ja,» sagte Fagin, «'s ist sehr schade, Bill, daß es eben nichts
gibt zu tun.»

«Da hast du mal recht», entgegnete Sikes barsch. «'s ist um so mehr
schade, da ich obendrein recht in der Laune dazu bin.»

Der Jude schüttelte seufzend den Kopf.

«Wir müssen die verlorene Zeit wieder einzubringen suchen, wenn wieder
was Gutes eingefädelt ist», fuhr Sikes fort.

«So ist's recht, mein Lieber», erwiderte Fagin, sich erdreistend, ihn
auf die Schulter zu klopfen. «Es freut mich herzinnig, Euch reden zu
hören so.»

«Freut Euch herzinnig -- so! Meinetwegen», sagte Sikes.

«Ha, ha, ha!» lachte der Jude, als wenn ihm schon dies sehr geringe
Zugeständnis Freude gewährte. «Ihr seid heute abend der echte,
wahrhaftige Bill -- wieder ganz Ihr selber, mein Lieber.»

«Mir ist's, als wär ich ein ganz anderer, wenn du mir die alte, welke
Tatze auf die Schulter legst -- runter damit!» rief Sikes, die Hand des
Juden zurückschleudernd.

«Wird Euch schlimm dabei, Bill -- erinnert's Euch ans Gefaßtwerden?»
fragte der Jude, entschlossen, keine Empfindlichkeit zu zeigen.

«Ja -- aber ans Gefaßtwerden vom Teufel, nicht von 'nem Häscher. Von
Adam her ist kein Mensch gewesen mit 'nem Gesicht wie das deinige,
müßte denn sein dein Vater, und dem wird wohl jetzund sein grauer Bart
versengt, sofern nicht Satan selber dein Vater ist, was mich eben nicht
wundern würde.»

Fagin erwiderte nichts auf diese Schmeichelei, sondern zupfte Sikes
am Ärmel und wies nach Nancy hin, die ganz in der Stille den Hut
aufgesetzt hatte und eben hinausgehen wollte.

«Heda, Nancy!» rief Sikes. «Wohin will die Dirne bei dieser
Nachtstunde?»

«Nicht weit.»

«Was ist das für 'ne Antwort! Wohin willst du?»

«Ich sage, nicht weit.»

«Und ich sage, wohin? Hast du gehört?»

«Ich weiß nicht, wohin.»

«Dann weiß ich's», sagte Sikes, mehr aus Eigensinn, als daß er einen
bestimmten Grund gehabt hätte, sich Nancys Ausgehen, wohin es ihr
beliebte, zu widersetzen. «Nirgend. Setz dich wieder hin.»

«Ich bin unwohl, wie ich Euch schon gesagt habe, und muß frische Luft
schöpfen.»

«Steck den Kopf aus 'm Fenster 'naus, das ist ebensogut.»

«Das ist's nicht; ich muß Bewegung haben.»

«So -- du sollst aber keine haben», entgegnete Sikes, stand auf,
verschloß die Tür, zog den Schlüssel aus, riß dem Mädchen den Hut vom
Kopfe und warf ihn auf einen alten Schrank. «Willst du jetzt ruhig
dableiben, wo du bist, oder nicht?»

«Ich kann auch ohne Hut gehen», sagte Nancy erblassend. «Was soll dies
bedeuten, Bill? Wißt Ihr auch, was Ihr tut?»

«Ob ich weiß, was -- Fagin, sie ist von Sinnen, denn sie würde sich's
sonst nicht herausnehmen, solche Worte zu mir zu sprechen!»

«Ihr macht's danach, daß ich etwas Verzweifeltes tue», murmelte Nancy,
beide Hände gegen die Brust pressend, als wenn sie einen heftigen
Ausbruch gewaltsam zurückdrängen wollte. «Laßt mich hinaus -- in dieser
Minute -- diesem Augenblick --»

«Nein!» schrie Sikes.

«Fagin, sagt ihm, daß er mich gehen läßt. Ich rat's ihm. Hört Ihr?»
rief Nancy, mit den Füßen stampfend.

«Ob ich dich höre? Ja», rief Sikes zurück; «und wenn ich dich noch
ein paar Augenblicke höre, so soll dich der Hund dermaßen an der Kehle
packen, daß er dir die kreischende Stimme herausreißt. Was fällt dir
ein, Weibsbild -- was steckt dir im Kopfe?»

«Laßt mich gehen», sagte Nancy flehend, setzte sich an die Tür auf den
Boden nieder und fuhr fort: «Bill, laßt mich gehen; Ihr wißt nicht, was
Ihr tut -- wißt's wahrlich nicht. Nur eine -- nur eine einzige Stunde.»

«Ich will mich vierteln lassen,» rief Sikes, sie sehr unsanft beim Arm
fassend, «wenn ich nicht glaube, daß die Dirne verrückt -- toll und
verrückt geworden ist. Steh auf!»

«Ich stehe nicht eher auf, als bis Ihr mich gehen laßt -- nicht eher!»
schrie Nancy.

Sikes blickte sie eine Weile an, ersah den rechten Augenblick, faßte
plötzlich ihre beiden Hände, zog die Sträubende in ein anstoßendes
Gemach, setzte sich auf eine Bank, warf sie auf einen Stuhl und hielt
sie gewaltsam nieder. Sie bat und suchte sich ihm abwechselnd mit
Gewalt zu entziehen, gab endlich, als es zwölf geschlagen hatte, ganz
erschöpft ihre Versuche auf, und Sikes verließ sie mit einer durch
mehrfache kräftige Beteuerungen unterstützten Warnung, um zu Fagin
zurückzukehren.

«Was für'n sonderbares Geschöpf die Dirne ist!» sagte er, sich den
Schweiß abwischend.

«Das mögt Ihr wohl sagen -- mögt Ihr wohl sagen, Bill», versetzte der
Jude nachdenklich.

«Was meinst du denn, was ihr im Kopfe gesteckt hat, noch so spät mit
Gewalt ausgehen zu wollen? Du mußt sie besser kennen als ich -- was
meinst du, Jude?»

«Eigensinn, glaub' ich -- Weibertrotz und Eigensinn, mein Lieber»,
antwortete Fagin achselzuckend.

«Glaub's auch. Ich dachte, daß ich sie zahm gemacht hätte, sie ist aber
so schlimm wie je.»

«Noch schlimmer, Bill. Ich habe so etwas erlebt noch niemals an ihr,
und um solch 'ner geringen Ursache.»

«Ich auch nicht. Es scheint, mein Fieber steckt ihr im Blut und will
nicht 'raus -- was?»

«Mag wohl sein, Bill.»

«Ich will ihr 'n bissel Blut abzapfen, ohn' den Doktor zu bemühn, wenn
sie's wieder so macht.»

Der Jude nickte beistimmend.

«Sie war Tag und Nacht um mich,» fuhr Sikes fort, «als ich auf der
Seite lag, während du wie 'n falscher Kujon, der du bist, dich fern
hältst. Wir hatten die ganze Zeit nichts zu beißen und zu brechen,
und ich glaub', es hat sie verdrießlich gemacht, und sie ist unruhig
geworden, weil sie so lang hat im Haus sitzen müssen -- he?»

«Ganz recht, mein Lieber», erwiderte Fagin flüsternd. «Pst!»

In diesem Augenblick trat Nancy wieder herein und setzte sich an ihren
gewohnten Platz. Ihre Augen waren rot und geschwollen: sie wiegte sich
hin und her, warf den Kopf empor und brach nach einiger Zeit in ein
Gelächter aus.

«Was ist denn dies nun wieder?» rief Sikes, erstaunt zu Fagin sich
wendend, aus.

Der Jude gab ihm einen Wink, sie für den Augenblick nicht weiter zu
beachten, und nach einigen Minuten saß sie wieder da wie vorhin. Er
flüsterte Sikes zu, sie würde von nun an ganz ruhig bleiben, nahm
seinen Hut und sagte ihm gute Nacht. An der Tür stand er still, drehte
sich noch einmal um und bat, daß ihm jemand auf der dunkeln Treppe
leuchten möchte.

«Leucht ihm 'nunter», sagte Sikes, der eben seine Pfeife füllte.
«'s wäre schade, wenn er hier selbst den Hals bräche und den
Hängezuschauern nichts zu gaffen gäbe.»

Nancy geleitete den alten Mann mit dem Lichte hinunter. Auf dem
Hausflur angelangt, legte er den Finger auf den Mund und flüsterte ihr
in das Ohr: «Was hattest du, liebes Kind?»

«Wieso?» erwiderte sie, gleichfalls flüsternd.

«Warum du ausgehen wolltest mit Gewalt. Wenn er,» sagte Fagin, mit dem
knöchernen Finger nach oben zeigend, «wenn er ist so barbarisch gegen
dich -- er ist ein Tier, Nancy, ein unvernünftiges, wildes Tier --
warum --»

«Nun?» fragte sie, als er, den Mund dicht an ihrem Ohre und die Augen
dicht vor den ihrigen, innehielt.

«Laß jetzt gut sein,» fuhr der Jude fort, «wollen ein andermal sprechen
davon. Du hast einen Freund an mir, Kind, einen treuen Freund. Ich
hab' auch die Mittel -- wenn du willst dich rächen an ihm, der
dich behandelt wie einen Hund -- schlimmer als einen Hund, dem er
schmeichelt bisweilen doch -- so komm zu mir; komm zu mir, was ich dir
sage. Er ist ein Tagesfreund; aber mich kennst du von alters her, Nancy
-- von alters her.»

«Ich kenne Euch sehr wohl», antwortete das Mädchen, ohne die mindeste
Bewegung zu zeigen. «Gute Nacht.»

Sie trat zurück, als er ihr die Hand reichen wollte, sagte ihm aber
noch einmal mit fester Stimme gute Nacht, erwiderte den Blick, den er
ihr zum Abschiede zuwarf, mit einem hinlängliches Verstehen andeutenden
Zunicken und verschloß die Tür hinter ihm.

Fagin kehrte gedankenvoll nach seiner Wohnung zurück. Er war schon
seit einiger Zeit der Ansicht, in welcher ihn das soeben Vorgefallene
bestärkte, daß Nancy der schlechten Behandlung, welche sie von dem
brutalen Sikes erfuhr, müde geworden sei und eine Neigung zu einem
neuen Freunde gefaßt habe. Ihr verändertes Wesen, daß sie so häufig
allein ausging, ihre verhältnismäßige Gleichgültigkeit gegen den
Vorteil oder Schaden der Bande, für welche sie vormals so großen Eifer
bewiesen hatte, und dazu ihr so heftiges Verlangen, an diesem Abende
und gerade zu einer bestimmten Stunde das Haus noch verlassen zu
wollen: dieses alles unterstützte seine Annahme und überzeugte ihn fest
von der Richtigkeit derselben. Der Gegenstand dieser neuen Liebschaft
des Mädchens befand sich unter den Leuten seines Anhangs nicht. Er
mußte mit einer Alliierten wie Nancy eine schätzbare Erwerbung sein,
die es so bald wie möglich zu machen galt.

Auch war noch ein anderer und finsterer Zweck zu erreichen. Sikes
wußte zu viel, und seine plumpen, beleidigenden Reden hatten Fagin
darum nicht minder verletzt und gereizt, weil er es sich nicht merken
ließ. Nancy konnte es nicht entgehen, daß sie, sobald sie sich von
ihm trennte, vor seiner Wut nicht sicher war und daß er dieselbe ohne
allen Zweifel auch an ihrem neuen Liebhaber auslassen würde, so daß
die gesunden Gliedmaßen, ja das Leben desselben in offenbarer Gefahr
schwebten. Fagin glaubte, sie würde sich leicht bereden lassen, ihn zu
vergiften. «Weiber,» dachte er, «haben so etwas und noch Schlimmeres
wohl schon getan, um die Ziele zu erreichen, die das Mädchen jetzt
verfolgt. Tut sie es, so werde ich von dem gefährlichen Halunken,
dem Menschen, den ich hasse, befreit -- erhalte einen Ersatzmann für
ihn, und mein Einfluß über Nancy ist, bei meiner Kenntnis dieses
Verbrechens, fortan ganz unbegrenzt.»

Dies waren seine Gedanken gewesen, während ihn Sikes allein gelassen,
und er hatte deshalb beim Fortgehen das Mädchen auszuforschen gesucht.
Sie hatte keine Überraschung gezeigt, sich nicht angestellt, als ob sie
ihn nicht verstände, vielleicht bewies der Blick, mit welchem sie ihm
zum zweitenmal gute Nacht gesagt, klar, daß sie seine Meinung sehr wohl
verstanden hatte.

Aber sie weigerte sich vielleicht, in einen Anschlag auf Sikes Leben
einzugehen, worauf es hauptsächlich ankam. «Wie kann ich meinen Einfluß
bei ihr vergrößern?» dachte der Jude auf seinem Heimwege. «Welche neue
Gewalt über sie kann ich mir verschaffen?»

Ein Gehirn, wie das seinige, ist fruchtbar an Hilfsmitteln. Sollte er
sie nicht seinen Plänen fügsam machen können, wenn er sie, sofern kein
Geständnis von ihr zu erlangen war, von einem Kundschafter beobachten
ließ, den Gegenstand ihrer neuen Leidenschaft entdeckte und Sikes (vor
dem sie sich in hohem Maße fürchtete) alles zu enthüllen drohte, falls
sie nicht einwilligte, zu tun, was er von ihr verlangte.

«Es wird angehen», sagte er fast laut; «hab' ich nur erst ihr
Geheimnis, so darf sie mir's nicht abschlagen -- so gewiß ihr an ihrem
Leben liegt. Ich besitze die Mittel, Nancy, und nur Geduld, Sikes, ich
hab' euch, hab' euch beide!»

Er wandte sich mit einer drohenden Handbewegung um und warf einen
finsteren Blick nach der Straße zurück, wo er den verwegenen Bösewicht
verlassen, und senkte im Weitergehen die knöchernen Hände in die Falten
seines zerlumpten Mantels, die er zusammenpreßte, als wenn er einen
verhaßten Feind zwischen den spitzigen Fingern hätte.




45. Kapitel.

    Noah Claypole wird von Fagin als Spion verwandt.


Der alte Mann stand am anderen Morgen beizeiten auf und erwartete
ungeduldig seinen neuen Verbündeten, der sich erst nach einem endlos
scheinenden Ausbleiben zeigte und sogleich mit Gier über das Frühstück
herfiel.

«Bolter», sagte der Jude, sich ihm gegenüber setzend.

«Was gibt's?» erwiderte Noah. «Fordert nichts von mir, bis ich mit'm
Essen fertig bin. Das ist der große Fehler hier. Es wird einem niemals
Zeit genug bei den Mahlzeiten gelassen.»

«Ei, Ihr könnt doch sprechen beim Essen», sagte der Jude, vom Grunde
seines Herzens des jungen Freundes Eßgier verwünschend.

«Und es geht obendrein noch besser, wenn ich spreche», versetzte
Bolter, ein ungeheures Stück Brot abschneidend. «Wo steckt denn
Charlotte?»

«Ich habe sie ausgeschickt heute morgen mit dem anderen jungen
Frauenzimmer, weil ich wünschte zu sein allein.»

«Wollte nur, daß Ihr der Dirne erst gesagt hättet, sie sollte
Brotschnitte mit Butter rösten. Nun schwatzt aber nur zu -- werde mich
nicht stören lassen», sagte Noah, und es schien in der Tat wenig auf
sich zu haben mit der Besorgnis, daß er sich stören lassen dürfte, denn
er war offenbar entschlossen, wacker fortzuarbeiten.

«Ihr habt gestern gemacht Eure Sachen gut», sagte der Jude; «sehr
schön. Sechs Schillinge, neun Pence und 'nen halben Penny am
allerersten Tage! Das Schratzchen wird Euch machen reich.»

«Vergeßt nicht die drei Bierkannen und den Milchtopf», erwiderte Bolter.

«Nein, nein, mein Lieber. Die Bierkannen waren große Geniebeweise, der
Milchtopf aber war ein vollkommenes Meisterstück.»

«Ging wohl an für 'nen Anfänger», bemerkte Mr. Bolter selbstgefällig.
«Die Bierkannen nahm ich von 'nem Sout'raingitter 'runter, und der
Milchtopf stand draußen vor 'nem Gasthofe; ich dachte also, er möchte
rostig werden durch den Regen oder sich erkälten, wißt Ihr. Ha, ha, ha!»

Der Jude stimmte in Mr. Bolters Gelächter, der seine Beschäftigung
rüstig weiter fortsetzte, des Scheines halber herzlich ein und sagte,
sich über den Tisch hinüberlehnend: «Ihr müßt mir ausrichten etwas,
mein Lieber, das erfordert große Sorgfalt und Vorsicht.»

«Fagin,» entgegnete Noah, «Ihr dürft aber nichts Gefährliches von mir
verlangen und mich nicht wieder in Polizeigerichte schicken; denn ein
für allemal, das gefällt mir nicht, und ich will's nicht.»

«'s ist dabei nicht die geringste Gefahr -- Ihr sollt bloß baldowern
ein Frauenzimmer.»

«Ein altes?»

«Ein junges.»

«Nun, darauf versteh' ich mich gut genug -- trieb's schon mit Glück,
als ich noch in die Schule ging. Was soll ich denn auskundschaften von
der jungen Person?»

«Wohin sie geht, mit wem sie verkehrt, und womöglich, was sie sagt;
Euch merken die Straße, wenn's eine Straße, oder das Haus, wenn's ist
ein Haus, und mir bringen so viel Kunde, wie Ihr nur vermögt.»

«Was gebt Ihr mir dafür?» fragte Noah begierig.

«Wenn Ihr's gut ausrichtet, ein Pfund, mein Lieber -- ja, ja, ein
Pfund», erwiderte Fagin, der ihn so sehr wie möglich für die Sache zu
interessieren wünschte; «und das ist so viel, als ich noch nie habe
gegeben für ein Stück Arbeit, wobei nicht war viel zu gewinnen.»

«Wer ist denn das Frauenzimmer?»

«Eine der Unsern.»

«Hm -- so! Ihr setzt Mißtrauen in sie?»

«Sie hat sich gewandt zu neuen Liebhabern, und ich muß wissen, wer die
mögen sein.»

«Verstehe schon -- um das Vergnügen zu haben, sie kennen zu lernen,
wenn es respektable Leute sind -- wie? Ha, ha, ha! Verlaßt Euch auf
mich.»

«Wußte wohl, daß ich's würde können.»

«Natürlich, natürlich! Wo ist sie? Wo muß ich ihr auflauern? Wann geh'
ich los?»

«Ihr sollt das alles hören von mir, mein Lieber, zu seiner Zeit. Haltet
Euch bereit nur und überlaßt das übrige mir.»

Sechs Abende saß der Kundschafter gestiefelt und in seinem
Kärrneranzuge da, bereit, auf einen Wink von Fagin zu beginnen, und
Abend für Abend kehrte der Jude verdrießlich nach Hause zurück und
sagte, daß es noch nicht Zeit wäre. Am siebenten -- einem Sonntagabend
-- trat er mit einem Vergnügen ein, das er nicht zu verbergen imstande
war.

«Sie geht aus heute abend,» sagte er, «und ich bin gewiß, daß sie geht
hin da, wo ist zu erforschen, was ich wünsche zu wissen; denn sie
hat allein gesessen den ganzen Tag, und der Mann, vor dem sie sich
fürchtet, wird erst zurückkehren gegen Tagesanbruch. Kommt, kommt!
Folgt mir geschwind!»

Des Juden Erregtheit steckte auch Noah an, der sogleich aufsprang. Sie
verließen das Haus, eilten durch ein Straßen- und Gassenlabyrinth und
langten endlich vor einem Gasthause an, in welchem Noah die Krüppel
erkannte. Es war elf Uhr vorüber und die Tür verschlossen; sie öffnete
sich aber auf ein leises Pfeifen des Juden und schloß sich wieder, als
sie geräuschlos hineingegangen waren. Fagin flüsterte kaum, sondern
besprach sich mit dem jüdischen Jünglinge durch stumme Zeichen, wies
darauf nach dem kleinen Fenster hin und bedeutete Noah, auf einen
Stuhl zu steigen und sich die im anstoßenden Zimmer befindliche Person
anzusehen.

«Ist das das Frauenzimmer?» flüsterte Noah. «Sie sieht vor sich nieder,
und das Licht steht hinter ihr. Ich kann ihr Gesicht nicht erkennen.»

«Bleibt ruhig stehen», flüsterte Fagin und gab Barney ein Zeichen, der
sogleich hinausging, nach ein paar Augenblicken in dem anstoßenden
Zimmer erschien, das Licht, unter dem Vorwande, es zu schneuzen, vor
das Frauenzimmer -- Nancy -- hinstellte, sie anredete und dadurch
veranlaßte, den Kopf emporzuheben.

«Jetzt seh' ich sie», flüsterte Noah.

«Deutlich?»

«Würde sie unter Tausenden wiedererkennen.»

Nancy stand auf und schickte sich zum Fortgehen an. Er stieg eilig von
dem Stuhle herunter und trat sacht mit Fagin hinter einen Vorhang;
gleich darauf ging Nancy durch das Zimmer und aus dem Hause hinaus.

«Pst!» rief Barney, der ihr die Haustür geöffnet, «jetzt.»

Noah wechselte einen Blick mit Fagin und schlüpfte hinaus.

«Links», flüsterte Barney; «haltet Euch linker Hand und auf der anderen
Seite.»

Noah sah Nancy beim Laternenscheine schon in einiger Entfernung. Er
eilte ihr nach, folgte ihr so nahe, wie es ihm rätlich erschien, und
hielt sich auf der anderen Seite, um sie desto besser beobachten zu
können. Sie sah sich ängstlich ein paarmal um und stand einmal still,
um einige ihr dicht nachfolgende Männer vorüberzulassen. Sie schien im
Weitergehen Mut zu gewinnen und einen sicheren und festeren Schritt
anzunehmen. Der Kundschafter hielt sich in gemessener Entfernung hinter
ihr und ließ sie nicht aus den Augen.




46. Kapitel.

    Nancy erfüllt ihre Zusage.


Die Kirchenglocken schlugen dreiviertel auf elf Uhr, als zwei Gestalten
auf der Londoner Brücke erschienen. Die eine leicht und rasch vorwärts
eilende war die eines Mädchens, das unruhig um sich blickte, als
erwarte sie jemand; die andere die eines Mannes, der im tiefsten
Schatten, den er finden konnte, der ersteren in einiger Entfernung
nachschlich, aber stillstand, wenn das Mädchen stillstand, und wieder
vordrang, so schnell oder langsam dasselbe sich eben fortbewegte. So
schritten sie über die Brücke von dem Middlesex- nach dem Surreyufer
hinüber. Das Mädchen, das alle Vorübergehenden mit forschenden Blicken
gemustert hatte, schien sich in seiner Erwartung getäuscht zu haben,
drehte sich plötzlich um und ging wieder zurück. Der Kundschafter war
indes auf seiner Hut gewesen, trat in eine Vertiefung, lehnte über das
Geländer, ließ das Mädchen vorüber und folgte ihr sodann wieder nach
wie vorher. Fast mitten auf der Brücke angelangt, stand sie still und
er gleichfalls.

Es war eine sehr finstere und kalte Nacht, nur wenige gingen an den
beiden vorüber und beachteten sie nicht. Die Themse war von dichtem
Nebel bedeckt, den der matte, rötliche Glanz der Feuer auf den
kleinen, in den Werften ankernden Fahrzeugen kaum zu durchdringen
vermochte, und die Feuer ließen die Häuser am Ufer nur als dämmrige,
noch undeutlichere Massen erscheinen. Die Türme der alten Heilands-
und St.-Magnus-Kirche -- so lange schon die riesigen Wächter der
alten Brücke -- waren sichtbar durch die Finsternis, der Wald der
Schiffsmaste aber unter der Brücke und weiter umher die Menge der Türme
auch für den schärfsten Blick unerkennbar.

Das Mädchen war -- fortwährend von seinem ungesehenen Beobachter
verfolgt -- unruhig ein paarmal hin und wieder über die Brücke
gegangen, als die Glocke der St.-Pauls-Kirche abermals das
Hinscheiden eines Tages verkündete. Mitternacht war gekommen über
die menschenerfüllte Stadt, die Paläste und Hütten, die Bettler- und
Gaunerhöhlen, den Kerker und das Irrenhaus, die Gemächer, in welchen
neues Leben begann und abgelaufenes endete, Gesunde ruhten und Kranke
ächzten, Leichen starr dalagen und blühende Kinder süß schlummerten und
träumten.

Nicht zwei Minuten, nachdem der letzte Glockenton verklungen war,
stiegen eine junge Dame und ein grauköpfiger Herr aus einem Mietswagen
nicht weit von der Brücke, auf welche sie rasch zuschritten. Sie hatten
sich kaum auf derselben gezeigt, als das Mädchen aufmerksam stillstand
und ihnen sodann entgegeneilte, deren Munde ein Ausruf der Überraschung
entfloh, welchen sie jedoch sogleich unterdrückten, als ein wie ein
Kärrner Gekleideter plötzlich fast gegen sie anrannte.

«Nicht hier», flüsterte Nancy hastig. «Ich fürchte mich, hier mit Ihnen
zu reden. Folgen Sie mir dort die Treppe hinunter.»

Der Kärrner drehte sich um, während sie so sprach und nach der Treppe
hinwies, rief in rauhem Tone zurück, «wozu sie die Breite des ganzen
Steinpflasters einnähmen», und ging vorüber.

Die Treppe, nach welcher das Mädchen hingewiesen hatte, befand sich am
Surreyufer und führte zu einem Landungsplatze hinunter; der Kärrner
eilte hin zu ihr, blickte forschend umher und fing an, hinabzusteigen.
Sie besteht aus drei Absätzen, auf deren zweitem die Mauer linker Hand
in einen Pfeiler nach der Themse hin ausläuft. Die Stufen der unteren
Flucht sind breiter, und wer nur um eine einzige tiefer hinter den
Pfeiler tritt, ist denen verborgen, die, wenn auch ganz in seiner Nähe,
auf dem Treppenabsatze stehen. An dieser Stelle versteckte sich der
Kärrner, mit dem Rücken an den Pfeiler tretend. Er war in gespanntester
Erwartung, denn was hier vorging, lag gänzlich außer dem Kreise aller
seiner Vermutungen, und wollte schon wieder höher hinaufgehen, als er
den Schall von Fußtritten und gleich darauf dicht neben sich Stimmen
vernahm. Er horchte mit verhaltenem Atem.

«Dies ist weit genug», sagte der Herr. «Ich lasse die junge Dame nicht
weiter hinuntergehen. Viele andere würden Ihnen nicht einmal so weit
gefolgt sein; Sie sehen, daß ich Ihnen Vertrauen bewiesen habe.»

«Sie sind in der Tat sehr vorsichtig -- oder auch mißtrauisch, wie mir
scheint. Doch gleichviel», sagte Nancy.

«Weshalb führen Sie uns denn aber an einen solchen Ort?» fragte der
Herr in einem milderen Tone. «Warum wollten Sie sich nicht dort oben
sprechen lassen, wo es hell ist und wo doch Menschen in der Nähe sind?»

«Ich habe es Ihnen schon gesagt, daß ich mich fürchtete, dort mit
Ihnen zu reden. Ich weiß nicht, wie es kommt,» sagte Nancy schaudernd,
«bin aber so beklommen und zittere so sehr, daß ich kaum auf den Füßen
stehen kann.»

«Was fürchten Sie denn?» fragte der Herr mitleidig.

«Ich weiß es selbst kaum», erwiderte das Mädchen. «Den ganzen Tag
haben mich schreckliche Gedanken an die verschiedensten Todesarten und
blutige Leichentücher heimgesucht, und fortwährend hat mich eine Angst
gequält, daß es mir war, als wenn ich mitten im Feuer brannte. Ich las
heute abend in einem Buche, um mir die Zeit zu verkürzen, und las nur
immer dasselbe heraus.»

«Einbildungen», sagte der alte Herr beruhigend.

«Nein, nein», entgegnete das Mädchen mit heiserer Stimme. «Ich will
darauf schwören, daß das Wort >Sarg< auf jeder Seite des Buches mit
großen, schwarzen Lettern gedruckt stand -- und erst vor kurzem, als
ich hierher ging, ward einer dicht an mir vorübergetragen.»

«Das ist nichts Ungewöhnliches. Ich habe sehr oft Särge an mir
vorübertragen sehen.»

«*Wirkliche* -- das war aber dieser nicht.»

Sie sprach dies alles in einem Tone, daß es den versteckten Lauscher
kalt überlief, ja, daß ihm das Blut in den Adern erstarrte. Er
hatte nie eine größere Herzenserleichterung empfunden, als in dem
Augenblicke, da er die süße Stimme der jungen Dame -- Roses -- vernahm,
die Nancy bat, sich zu beruhigen und sich nicht so entsetzlichen
Gedanken hinzugeben.

«Reden Sie ihr freundlich zu, der Armen; sie scheint es zu bedürfen»,
fügte sie, zu ihrem Begleiter sich wendend, hinzu.

«Ihre hochmütigen, frommen Damen würden mich, wenn sie mich in dieser
Nacht sähen, wie ich bin, verächtlich anblicken und mir vom ewigen
Höllenfeuer und der Rache des Himmels predigen», rief das Mädchen
aus. «Oh, meine teure junge Dame, warum sind die nicht, die Gottes
Auserwählte sein wollen, so mild und gütig gegen uns arme Unglückliche
wie Sie! Ach, Sie besitzen alles, was jene verloren haben, Jugend und
Schönheit, und könnten gar wohl ein wenig stolz sein, statt so viel
bescheidener.»

«Ah!» fiel der Herr ein; «ein Türke kehrt sein Antlitz, nachdem er es
reinlich abgewaschen, nach Osten, indem er seine Gebete spricht. Jene
guten Leute reiben an der rauhen Welt die Freundlichkeit von ihren
Gesichtern ab und wenden sie dann nach der finsteren Seite des Himmels.
Hab' ich zwischen dem Muselman und Pharisäer zu wählen, so lobe ich mir
den ersteren.»

Er sprach die Worte zu der jungen Dame, doch vielleicht beabsichtigend,
Nancy Zeit zu verschaffen, sich wieder zu sammeln. Bald darauf redete
er das Mädchen an.

«Sie waren am vorigen Sonntagabend nicht hier.»

«Ich konnte nicht kommen -- wurde gewaltsam zurückgehalten.»

«Von wem?»

«Von Bill -- dem Manne, von dem ich der jungen Dame erzählt habe.»

«Ich will doch hoffen, daß niemand Verdacht wegen der Sache auf Sie
geworfen hat, die uns jetzt zusammengeführt?» fragte der alte Herr
besorgt.

«Nein», antwortete Nancy kopfschüttelnd. «Es ist aber nicht eben leicht
für mich, ihn zu verlassen, ohne daß er weiß, warum, und ich würde auch
zu der Dame nicht haben gehen können, hätt' ich ihm nicht, um mich von
ihm entfernen zu können, einen Schlaftrunk gegeben.»

«War er denn vor Ihrer Rückkehr erwacht?»

«Nein; und so wenig, wie er selbst, hat sonst jemand Verdacht auf mich
geworfen.»

«Gut. Hören Sie mich jetzt an. Diese junge Dame hat mir und einigen
andern, das vollkommenste Zutrauen verdienenden Freunden mitgeteilt,
was Sie vor vierzehn Tagen ihr anvertraut haben. Ich gestehe, anfangs
Zweifel gehegt zu haben, ob man sich ganz auf Ihre Aussagen verlassen
könnte, halte mich aber jetzt davon überzeugt.»

«Das können Sie allerdings sein», beteuerte Nancy.

«Ich wiederhole, daß ich es fest glaube; und um Ihnen zu beweisen, daß
ich Ihnen zu vertrauen geneigt bin, sage ich Ihnen ohne Rückhalt, daß
wir das Geheimnis, worin es auch bestehen mag, Monks durch Furcht zu
entreißen gesonnen sind. Doch wenn -- wenn wir seiner nicht sollten
habhaft werden können, oder wenn ihm nichts abzudringen wäre, so müssen
Sie uns den Juden in die Hände liefern.»

«Fagin!» rief das Mädchen, plötzlich zurücktretend, aus.

«Ihn -- ja, ihn müssen Sie uns in die Hände liefern.»

«Nimmermehr -- das werd' ich nimmermehr tun», entgegnete Nancy; «werde
es nie tun, solch ein Teufel er auch ist, und obwohl er ärger als ein
Teufel an mir gehandelt hat.»

«Sie wollen also nicht?» fragte der Herr, der keine andere Antwort
erwartet zu haben schien.

«In keinem Falle!»

«Dann sagen Sie mir, warum Sie es nicht wollen.»

«Aus einem Grunde,» erwiderte Nancy mit Festigkeit, «aus einem Grunde,
den die Dame kennt, und ich weiß, denn ich habe ihr Versprechen, sie
wird dabei auf meiner Seite stehen; und aus dem weiteren Grunde, weil
ich -- ein so ruchloses Leben er auch geführt hat -- gleichfalls einen
schlechten Wandel geführt habe. Viele von uns sind miteinander schlecht
und böse gewesen, und ich will sie nicht verraten, die mich hätten
verraten können und es, so schlecht sie sind, nicht taten.»

«Dann,» fiel der Herr lebhaft ein, als wenn er erreicht hätte, was er
eben gewollt, «dann liefern Sie mir Monks in die Hände und überlassen
Sie es mir, nach Gutdünken mit ihm zu verfahren.»

«Wie aber, wenn er die andern verrät?»

«Ich verspreche Ihnen, daß die Sache ruhen soll, sobald wir ihm die
Wahrheit abgerungen haben. In Olivers kleiner Lebensgeschichte kommen
ohne Zweifel Umstände vor, die man nur sehr ungern der Öffentlichkeit
preisgeben würde, und ist nur die Wahrheit heraus, so soll niemand in
Ungelegenheit kommen.»

«Aber wenn Sie sie nicht herausbekommen?»

«Dann soll der Jude nicht ohne Ihre Einwilligung vor Gericht gezogen
werden, und ich glaube Ihnen für den Fall Gründe vorlegen zu können,
nach deren Anhören Sie einwilligen werden.»

«Hab' ich dafür das Versprechen der Dame?» fragte Nancy mit Nachdruck.

«Ich gebe es Ihnen», nahm Rose das Wort; «gebe Ihnen die aufrichtigste
und bestimmteste Zusage.»

«Monks soll nie erfahren, wie Sie zu der Kunde, die Sie durch
mich besitzen, gelangt sind?» fuhr das Mädchen nach einem kurzen
Stillschweigen fort.

«Nein -- nie,» erwiderte der Herr. «Er soll es nicht einmal vermuten
können.»

«Ich bin eine Lügnerin gewesen und habe unter Lügnern gelebt von meiner
frühsten Kindheit an, will aber Ihren Worten Glauben schenken», sagte
Nancy nach einem abermaligen Stillschweigen.

Beide versicherten ihr, daß sie es getrost könne, und nunmehr nannte
sie ihnen, so leise flüsternd, daß der Horcher nur sehr schwer zu
verstehen vermochte, den Namen, den Stadtteil und die Straße der
Taverne, aus welcher ihr Noah nach der Brücke gefolgt war. Sie sprach
in kurzen Pausen; der Herr schien sich das Nötigste zu notieren. Als
sie auch das Innere des Hauses beschrieben und angegeben hatte, wie
es am besten beobachtet werden könnte, und an welchen Abenden und zu
welchen Stunden es von Monks besucht zu werden pflegte, schien sie ein
paar Augenblicke innezuhalten, um sich die Züge und das ganze Äußere
desselben um so lebhafter zurückzurufen.

«Er ist groß,» sagte sie, «und kräftig gebaut, aber nicht stark; er hat
einen lauernden Gang und blickt beim Gehen beständig erst über die eine
und dann über die andere Schulter. Vergessen Sie das nicht, denn seine
Augen liegen so tief wie nur immer möglich im Kopfe, so daß Sie ihn
daran fast allein schon unter Tausenden zu erkennen vermögen. Er hat
dunkles Haar und Augen und ein schwärzliches Gesicht, das aber ältlich
und verfallen aussieht, obwohl er nicht über sechs- bis achtundzwanzig
Jahre alt sein kann. Seine Lippen sind oft blau und durch Bisse
entstellt, denn er hat fürchterliche Zufälle und beißt sich bisweilen
sogar die Hände blutig -- warum stutzen Sie?» fragte sie plötzlich
abbrechend.

Der Herr erwiderte hastig, daß er sich dessen nicht bewußt wäre, und er
bat sie, fortzufahren.

«Ich mußte dies großenteils von andern herauslocken, um es Ihnen sagen
zu können,» sprach das Mädchen weiter, «denn ich habe ihn nur zweimal
gesehen, und beide Male war er in einen weiten Mantel eingehüllt. Mehr
glaube ich Ihnen nicht -- doch ja! An seinem Halse, so hoch hinauf, daß
man etwas davon sehen kann, wenn er sein Gesicht abwendet, ist --»

«Ein breites rotes Mal, wie von einer Brandwunde», fiel der Herr ein.

«Wie -- Sie kennen ihn?» rief das Mädchen aus.

Roses Lippen entfloh ein Ausruf des höchsten Erstaunens, und auf einige
Augenblicke waren alle drei so stumm, daß der Horcher sie atmen hören
konnte.

«Ich glaube es», unterbrach der Herr jedoch bald das Stillschweigen.
«Nach Ihrer Beschreibung sollte ich ihn allerdings kennen. Wir werden
indes sehen. Es gibt auffallende Ähnlichkeiten -- kann sein, daß er
dennoch ein anderer ist.»

Er trat bei diesen, in einem verstellt gleichgültigen Tone gesprochenen
Worten ein paar Schritte zurück, wobei er sich dem versteckten
Kundschafter näherte, der ihn flüstern hörte: «Er muß es sein!» Gleich
darauf sagte er wieder laut: «Junges Mädchen, Sie haben uns die
wichtigsten Dienste geleistet, und ich wünsche Ihnen dankbar dafür zu
sein. Was kann ich für Sie tun?»

«Nichts», erwiderte Nancy.

«So dürfen Sie nicht sprechen», fuhr der Herr in einem so dringenden
und herzlichen Tone fort, daß auch ein weit verhärteteres Gemüt dadurch
hätte gerührt werden mögen. «Ich bitte, sagen Sie es mir.»

«Ich muß dabei bleiben, Sir», entgegnete Nancy weinend. «Sie können
nichts tun, mir zu helfen. Für mich ist wahrlich keine Hoffnung übrig.»

«Sie schneiden sich die Hoffnung selbst ab», fuhr der Herr fort. «Ihre
Vergangenheit ist eine beklagenswerte Verschwendung unschätzbarer
Jugendgaben gewesen, wie sie der Schöpfer nur einmal gibt und nicht
wieder verleiht; auf die Zukunft aber können Sie Hoffnung setzen.
Ich sage nicht, daß es in unserer Macht stehe, Ihnen Seelenfrieden
zu bieten, der Ihnen nur in dem Maße werden kann, wie Sie selbst ihn
suchen; wohl aber sind wir imstande, und es ist unser eifriger Wunsch,
Ihnen einen stillen Zufluchtsort entweder im Lande oder, wenn Sie
Furcht hegen, hierzubleiben, außer Landes zu verschaffen. Noch ehe
der Morgen graut, sollen Sie Ihren bisherigen Genossen so gänzlich
entrückt sein und so wenige Spuren hinter sich zurücklassen, als wenn
Sie von der Erde verschwunden wären. Geben Sie unseren Vorstellungen
und Bitten nach. Ich möchte nicht, daß Sie auch nur noch ein einziges
Wort mit den Leuten Ihres bisherigen Umgangs wechselten, nur noch einen
Blick auf die Stätte Ihres bisherigen Daseins würfen, oder die Luft
nur wieder atmeten, welche Pest und Tod für Sie ist. Geben Sie unsern
Bitten nach, während es noch Zeit ist, solange Sie noch können.»

«Sie läßt sich bewegen», rief Rose aus; «ich weiß es, sie faßt den
rettenden Entschluß.»

«Nein, nein», erwiderte Nancy nach einem kurzen inneren Kampfe; «ich
bin an mein bisheriges Leben gekettet. Ich verabscheue und hasse es
jetzt, kann es aber nicht aufgeben. Ich war schon längst zu weit
gegangen, um zurückkehren zu können -- und doch weiß ich nicht, ob ich
es nicht versucht haben würde, wenn Sie vor einiger Zeit so zu mir
gesprochen hätten. Doch diese Angst ergreift mich wieder,» setzte sie,
sich hastig umwendend, hinzu: «ich muß nach Hause gehen.»

«Nach Hause?!» wiederholte Rose, großen Nachdruck auf die Worte legend.

«Nach Hause, Miß -- nach einem solchen Hause, wie ich es mir durch die
ganze Mühe meines Lebens erbaut habe. Lassen Sie uns scheiden. Man wird
mich beobachten oder sehen. Fort, fort von hier! Habe ich Ihnen einen
Dienst geleistet, so erzeigen Sie mir nur die einzige Güte, zu gehen
und mich allein nach Hause zurückkehren zu lassen.»

«Es ist vergeblich», sagte der Herr seufzend. «Wir gefährden vielleicht
Ihre Person, wenn wir hier weilen, und haben Sie vielleicht schon
länger aufgehalten, als Sie erwartet haben.»

«Ja, ja, das haben Sie», sagte Nancy.

«Was kann das Ende des Lebens der Ärmsten sein?» rief Rose aus.

«Schauen Sie hinunter in das finstere Wasser!» sagte das Mädchen.
«Wie oft lesen Sie von meinesgleichen, die sich in die Fluten
hinunterstürzen und kein lebendes Wesen, sie zu beweinen oder nur nach
ihnen zu fragen, zurücklassen. Es können Jahre darüber hingehen oder
vielleicht nur Monate, doch nicht besser wird zuletzt mein Ende sein.»

«O bitte, reden Sie nicht so», sagte Rose schluchzend.

«Sie werden nie davon hören, beste junge Dame, und Gott verhüte, daß
solcher Graus -- Gute Nacht, gute Nacht!»

Der Herr wandte sich ab.

«Nehmen Sie um meinetwillen diese Börse,» sagte Rose, «damit es Ihnen
in der Stunde der Not nicht an einer Hilfsquelle mangle.»

«Nein, nein», entgegnete das Mädchen. «Ich habe, was ich tat, nicht für
Geld getan. Lassen Sie mir dieses Bewußtsein. Doch -- geben Sie mir ein
Andenken -- etwas, das Sie getragen haben -- nein, nein, keinen Ring --
Ihre Handschuhe oder Ihr Taschentuch -- so, des Himmels Segen über Sie
-- gute Nacht, gute Nacht!»

Ihre heftige Erregtheit und die Besorgnis einer Entdeckung, welche
gefährlich für sie werden könnte, bewog den Herrn, sich ihrem Verlangen
gemäß mit Rose zu entfernen. Auf der obersten Stufe angelangt, standen
beide still.

«Horch!» flüsterte Rose. «Rief sie nicht? Ich glaube, daß ich ihre
Stimme hörte!»

«Nein, mein liebes Fräulein», erwiderte Brownlow, traurig
zurückblickend. «Sie hat sich nicht einmal leise geregt und wird es
auch nicht eher, als bis wir fort sind.»

Rose zögerte noch immer, allein der alte Herr legte ihren Arm in den
seinigen und zog sie mit sanfter Gewalt fort. Sobald sie verschwunden
waren, warf sich Nancy fast der Länge nach auf eine der Treppenstufen
nieder und machte ihrer Herzensqual durch bittere Tränen Luft. Nach
einiger Zeit stand sie wieder auf und begann mit wankenden Schritten
ihren Heimweg. Der erstaunte Horcher blieb noch einige Minuten hinter
dem Pfeiler stehen, schlich sodann die Treppe hinauf, lugte vorsichtig
umher und eilte dann, so schnell er konnte, nach dem Hause des Juden
zurück.




47. Kapitel.

    Unglückliche Folgen.


Es war fast zwei Stunden vor Tagesanbruch -- die rechte Nachtzeit im
Herbste, da, indem sogar der Schall zu schlummern scheint, die Straßen
schweigend und verlassen und die Schlemmer und Schwelger nach Hause
getaumelt sind, um zu träumen -- als der Jude wachend in seiner alten
Höhle mit einem so bleichen und verzerrten Gesichte und so roten und
blutunterlaufenen Augen dasaß, daß er weniger einem Menschen als einem
greulichen, vom Grabe feuchten und von einem bösen Geiste gepeinigten
Gespenste glich.

Er kauerte, in eine zerlumpte Bettdecke gehüllt, an einem kalten Herde
und hatte die Blicke auf ein dem Erlöschen nahes Licht gerichtet,
das auf einem Tische neben ihm stand. Die rechte Hand hielt er, wie
in Gedanken verloren, an die Lippen und kaute an seinen langen,
schwarzen Fingernägeln, so daß man in dem sonst zahnlosen Munde
einige Vorderzähne erblickte, die einem Hunde oder einer Ratte hätten
angehören können.

Auf einer Matratze am Boden ausgestreckt lag Noah Claypole in festem
Schlafe. Zwischen ihm und dem Lichte schweiften die zerstreuten
Blicke des alten Mannes bisweilen hin und wieder, in dessen Innerem,
einander drängend, unruhige Gedanken und stürmische Leidenschaften
wogten und wühlten -- bitterer Verdruß über das Mißlingen seines
gewinnverheißenden Planes, tödlicher Haß gegen das Mädchen, das
hinterlistig mit Fremden zu verkehren gewagt hatte, gänzliches
Mißtrauen in die Aufrichtigkeit ihrer Weigerung, ihn zu verraten,
Ingrimm darüber, sich an Sikes nicht rächen zu können, Furcht vor
Entdeckung, Verurteilung und Tod, die wildeste, durch das alles
entzündete Wut und neue Pläne der Arglist und schwärzesten Bosheit. Er
saß da, ohne auch nur im mindesten seine Stellung zu verändern oder
anscheinend die Zeit zu beachten, bis der Schall von Fußtritten auf der
Straße bei seinem feinen Gehör seine Aufmerksamkeit zu erregen schien.

«Endlich,» murmelte er, über die trocknen, fieberheißen Lippen mit der
Hand hinfahrend, «endlich!»

Die Glocke ertönte leise, er ging hinaus und kehrte bald darauf mit
einem Manne zurück, der bis an das Kinn vermummt war und ein Bündel
unter dem Arme trug. Es war Sikes.

«Da», sagte der verwegene Raubgesell, das Bündel auf den Tisch werfend.
«Mach draus, was du kannst. Es hat mir Mühe genug gekostet; ich meinte
schon vor drei Stunden hier sein zu können.»

Fagin verschloß das Bündel, setzte sich wieder, blieb stumm, blickte
jedoch nach Sikes scharf hinüber, und seine Lippen zitterten so heftig,
und sein Gesicht war infolge der in ihm wühlenden Leidenschaften
so verändert, daß der Dieb unwillkürlich sich zurücklehnte und ihn
bestürzt ansah.

«Was gibt's?» fuhr er auf. «Zu allen Teufeln, was siehst du mich so an?»

Der Jude hob die rechte Hand empor und schüttelte den bebenden
Zeigefinger; allein seine Bewegung war so heftig, daß er kein Wort
hervorzubringen imstande war.

«Gott verdamm' mich!» rief Sikes, in seine Brusttasche greifend, aus.
«Er ist verrückt geworden. Ich muß auf meiner Hut gegen ihn sein.»

«Nein, o nein», brachte Fagin endlich hervor. «Ihr -- Ihr seid's nicht,
Bill. Gegen Euch hab' ich nichts -- gar nichts, Bill.»

«Hm, 's ist auch ein Glück für einen von uns -- gleichviel für wen»,
sagte Sikes, ein Pistol absichtlich hervorziehend und in eine andere
Tasche steckend.

«Ich hab' Euch zu sagen was, Bill,» fuhr der Jude näher rückend fort,
«was Euch noch mehr wird erzürnen als mich.»

«So?!» entgegnete Sikes mit einer ungläubigen Miene. «Wenn's aber wahr
ist, so tu's Maul auf und mach' g'schwind, oder Nancy wird glauben, daß
ich verloren wär'.»

«Das hat sie schon ausgemacht bei sich selbst bestimmt genug!»

Sikes blickte ihn ungewiß an, streckte die mächtige Faust nach ihm
aus, schüttelte ihn und forderte ihn barsch und polternd auf, sich
deutlicher zu erklären.

«Denkt Euch,» sagte der Jude mit vor Wut fast erstickter Stimme,
«der Bursch da schliche nachts hinaus auf die Straßen, knüpfte
an Einverständnisse mit unsern schlimmsten Feinden, gäbe ihnen
Beschreibungen von uns und unsern verborgensten Schlupfwinkeln,
verriete unsere geheimsten Pläne und Taten, setzte auch hinzu noch viel
Lügen -- was dann, was dann?»

Sikes erklärte unter einer furchtbaren Verwünschung, er würde ihm
in einem solchen Falle den Schädel unter den eisernen Nägeln seiner
Stiefel zermalmen.

«Aber wie, wenn ich's täte!» schrie der Jude fast, «ich, der ich weiß
so viel und so viele kann bringen an den Galgen.»

«Weiß nicht,» erwiderte Sikes, bei dem bloßen Gedanken die Zähne
zusammenbeißend und erblassend. «Aber ich würd' im Kerker was tun, daß
sie mich in Eisen schlagen müßten, und stellten sie mich mit dir vor
Gericht, würd' ich dir vor den Richtern und allen den Kopf einschlagen.
Ich würd' ne solche Kraft haben,» murmelte er, den sehnigen Arm auf und
nieder schwingend, «daß ich ihn dir zu Brei schlagen könnte, als wenn
ein belad'ner Frachtwagen drüberhingegangen wär'.»

«Würdet Ihr tun das wirklich?»

«Ob ich's wohl tun würd'! Stell mich auf die Probe.»

«Wenn's aber getan hätte Charley oder der Baldowerer oder Bet oder --»

«Ist mir gleichviel wer», unterbrach Sikes ungeduldig. «Ich würd' ihn
bezahlen, möcht's sein, wer wollte.»

Fagin blickte ihn abermals scharf an, winkte ihm, zu schweigen,
beugte sich über den Schläfer und schüttelte denselben, während Sikes
verwundert und erwartungsvoll, die Hände auf die Knie stemmend, dasaß.

«Bolter, Bolter! Der arme Junge», sagte Fagin, mit einer Miene
emporblickend, in welcher die Vorahnung einer teuflischen Freude sich
ausdrückte. «Er wird müd' -- müde davon, daß er hat müssen wach sein
ihretwegen so lange -- ihr hat nachschleichen müssen noch so spät,
Bill.»

«Was willst du damit sagen?» fragte Sikes, sich zurücklehnend.

Der Jude antwortete nicht, setzte seine Bemühungen, Noah zu wecken,
fort, und es war ihm endlich einigermaßen gelungen.

«Erzähl' noch einmal -- daß der es hört auch», sagte er, nach Sikes
hinzeigend.

«Was soll ich erzählen?» fragte Noah, noch halb im Schlafe.

«Das von -- Nancy», antwortete der Jude, Sikes fest am Arme fassend,
wie um ihn zu verhindern, fortzueilen, bevor er genug gehört hätte, und
fragte darauf dem schlaftrunkenen Noah mit einer Wut, deren er nur mit
Mühe Herr zu bleiben vermochte, alles ab, was der Lauscher erhorcht
hatte.

«Und was sagte sie,» fragte er endlich mit wutschäumenden Lippen, «was
sagte sie vom vorigen Sonntage?»

«Der Herr fragte sie, warum sie nicht am vorigen Sonntage gekommen
wäre,» antwortete Noah, in welchem eine Ahnung davon auftauchte, wer
Sikes sein möchte; «und sie sagte, weil sie gewaltsam zurückgehalten
worden wäre von Bill, dem Manne, von dem sie ihnen schon gesagt hätte.»

«Was weiter von ihm?» rief der Jude. «Was sagte sie von ihm weiter?
Sag' ihm das, sag' ihm das!»

«Es wäre nicht leicht für sie,» fuhr Noah fort, «aus dem Hause zu
kommen, ohn' daß er wüßte, wohin sie ginge, und sie hätt' ihm daher,
als sie das erstemal zu der Dame gekommen wäre, 'nen Schlaftrunk
eingeben müssen -- ha, ha, ha.»

«Höll' und Teufel!» schrie Sikes, von dem Juden sich losreißend. «Laß
mich!»

Er stürzte wütend hinaus, Fagin rief und eilte ihm nach, würde ihn
jedoch nicht zurückgehalten haben, wenn die Haustür nicht verschlossen
gewesen wäre.

«Laß mich 'naus,» tobte er, «oder nimm dich in acht! Laß mich 'naus --
hörst du?»

«Ein Wort, Bill -- bloß ein einziges Wort», versetzte der Jude, die
Hand auf das Türschloß legend und mit verstellter Besorglichkeit: «Ihr
-- Ihr wollt doch nicht tun etwas zu -- zu Gewaltsames, Bill?»

Der Tag brach an, es war hell genug, als sie einander in das Gesicht
schauten, um deutlich sehen zu können, und in ihren Augen blitzte ein
Feuer, dessen Bedeutung nicht mißzuverstehen war.

«Ich meine», setzte Fagin hinzu, einsehend, daß Verstellung nicht
mehr möglich war, «nichts Gewaltsames, wodurch wir geraten könnten in
Gefahr. Fein listig, Bill, und seid nicht zu verwegen.»

Er hatte unterdes aufgeschlossen, Sikes antwortete nicht, riß die Tür
auf, stürzte hinaus und eilte, ohne rechts oder links zu schauen, ohne
eine Gesichtsmuskel zu bewegen oder ein zorniges Wort zu murmeln,
mit verbissenen Zähnen und trotzig-blutdürstiger Entschlossenheit
nach seiner Wohnung. Er ging mit leisen Schritten hinauf, öffnete und
verschloß die Tür seines Zimmers, stellte einen schweren Tisch gegen
sie und schob den Bettvorhang zurück.

Und da lag Nancy, halb angekleidet. Sie schreckte aus dem Schlaf empor.

«Steh auf», sagte er.

«Bist du es?» rief sie ihm, erfreut über seine Rückkehr, entgegen.

«Ja. Steh auf!»

Es brannte ein Licht -- er schleuderte es unter den Kaminrost. Sie
stand auf und ging nach dem Fenster, um den Vorhang aufzuziehen.

«Laß das», herrschte er ihr zu. «'s ist hell genug für das, was wir zu
tun haben.»

«Bill,» sagte sie bestürzt, «was seht Ihr mich so an?»

Er heftete eine kurze Weile schnaubend und mit wogender Brust die
Blicke auf sie, packte sie darauf beim Kopfe und der Kehle, zog sie in
die Mitte des Gemachs, warf einen einzigen Blick nach der Tür und legte
seine schwere Hand auf ihren Mund.

«Bill, Bill!» keuchte sie, in Todesangst unter seinem Griff sich
sträubend, «ich will nicht schreien -- nicht weinen -- hört mich --
sprecht doch nur -- sagt mir, was ich getan habe.»

«Weißt es selbst, du Satan in Dirnengestalt. Bist belauert gewesen
gestern abend; ich weiß jedes Wort, was du gesagt hast.»

«Oh, um der Liebe des Himmels willen,» rief sie, sich fest an ihn
anklammernd, «dann schont mein Leben, wie ich Eures geschont habe.
Bill, bester Bill, Ihr könnt mich ja nicht morden wollen. Bedenkt, was
ich gestern abend um Euretwillen aufgegeben habe. Ihr sollt Zeit haben,
es zu bedenken, Euch dies Verbrechen zu ersparen -- ich lasse Euch
nicht los, nimmermehr! Bill, Bill, um Gottes Barmherzigkeit, um Euret-
und um meinetwillen, besinnt Euch, eh' Ihr mein Blut vergießt. Bei
meiner sündigen Seele, ich bin Euch treu gewesen!»

Er suchte sich gewaltsam von ihr loszumachen, allein vergebens, sie
hielt mit der Kraft der Verzweiflung fest.

«Bill,» rief sie und bemühte sich, den Kopf auf seine Brust zu legen,
«der Herr und die liebe Dame boten mir einen Zufluchtsort außer Landes
an. Laßt mich noch einmal zu ihnen, daß ich sie auf den Knien anflehe,
Euch dieselbe Liebe und Güte zu erweisen, und dann laßt uns aus dieser
Höhle entfliehen und weit von hier ein besseres Leben anfangen und
unser voriges Leben, ausgenommen im Gebet, vergessen und uns nie
wiedersehen. Es ist zur Reue niemals zu spät. Sie sagten es mir -- ich
fühle es jetzt -- aber wir müssen Zeit -- ein wenig, ein wenig Zeit
haben!»

Er befreite einen seiner Arme und ergriff seine Pistole; doch so
wütend er war, der Gedanke, daß sogleich alles entdeckt werden würde,
wenn er Feuer gäbe, flog ihm durch den Sinn, und er schlug sie daher
mit aller Kraft, die er zu sammeln vermochte, zweimal auf das zu ihm
emporgehobene, das seinige fast berührende Gesicht.

Sie wankte und stürzte, fast erblindet von dem aus einer tiefen Wunde
in ihrer Stirn hervorströmenden Blute, zu Boden, richtete sich jedoch
mühsam wieder auf die Knie, zog ein weißes Tuch -- das ihr von Rose
geschenkte -- aus dem Busen und hielt es in den gefalteten Händen so
hoch, als es ihre schwachen Kräfte erlaubten, zum Himmel empor und
flehte um Erbarmen zu ihrem Schöpfer.

Sie war gräßlich anzuschauen. Der Mörder wankte zurück nach der Wand,
hielt die Hand vor die Augen, um sie nicht zu sehen, ergriff einen
schweren Knotenstock und schlug sie nieder.




48. Kapitel.

    Sikes' Flucht.


In der ganzen großen Hauptstadt war an diesem Morgen sicher keine so
greuliche, ruchlose Tat geschehen. Die Sonne -- die helle Sonne, die
nicht bloß Licht, sondern neues Leben, Hoffnung und rüstige Frische den
Menschen zurückbringt -- ging strahlend auf über der menschenerfüllten
Stadt und ergoß ihren Glanz durch kostbar bemalte Scheiben wie durch
papierverklebte Fenster und hinein in den himmelanstrebenden Dom wie
in die schlechteste, niedrigste Hütte. Sie erhellte auch das Gemach,
in welchem die ermordete Nancy lag. Sikes bemühte sich, dem Eindringen
ihres Lichtes zu wehren, jedoch vergeblich, und hatte das Mädchen beim
ungewissen Dämmerscheine des Morgens einen fürchterlichen Anblick
dargeboten, so war ihre blutige Gestalt bei voller Tageshelle noch
zehnmal unheimlicher und schauerlicher anzuschauen.

Sikes war aus Furcht nicht von der Stelle gewichen. Er hatte ein leises
Ächzen der jammervoll Daliegenden vernommen, ein Zucken ihrer Hand
gewahrt und aber- und abermals geschlagen, denn Schrecken und Angst
waren bei ihm zu der Erbitterung des Hasses hinzugekommen. Er warf eine
Decke über sie; doch es war noch fürchterlicher, im Geiste ihre Augen
zu schauen, nach ihm sich wenden und dann emporstarren zu sehen, als
wenn sie des Himmels Rache herabriefen. Er entfernte die Decke wieder,
und da lag der schreckliche Leichnam, aus dessen Wunden das Blut noch
langsam hervorquoll.

Er zündete Feuer an und steckte den Knotenstock hinein, an welchem
Haare der Ermordeten klebten, die er, trotz seiner Eisenfestigkeit,
mit Zagen von den Flammen ergreifen sah, und ließ ihn darin, bis er
zerbrach und zu Asche verbrannte. Er wusch sich und rieb seine Kleider
ab. Sie hatten Flecke, die nicht ausgehen wollten, und er schnitt die
Stücke heraus und verbrannte sie. Das ganze Gemach war blutbefleckt --
sogar die Füße des Hundes waren blutig.

Er hatte während dieser ganzen Zeit nicht nach der Leiche
zurückgesehen, nicht ein einziges Mal, und ging, den Hund mit sich
fortziehend, ohne hinzublicken nach der Tür, verschloß sie und verließ
das Haus. -- Er schritt quer über die Straße und schaute nach dem
Fenster hinauf, um sich zu überzeugen, daß von außen nichts zu sehen
wäre. Das Fenster war durch den Vorhang verhüllt, den sie aufziehen
wollte, um dem Lichte freien Zugang zu verschaffen, das sie aber nie
wiedersehen sollte. Ihre Leiche lag ganz in der Nähe -- er wußte es --
und wie hell die Sonne das Fenster erleuchtete!

Es war ihm jedoch Erleichterung, das Zimmer verlassen zu haben; er
pfiff dem Hunde und entfernte sich eiligen Schrittes. Er ging durch
Islington und über Highgate-Hill, ungewiß, wohin er sich wenden sollte,
hatte endlich Hampstead hinter sich gelassen, befand sich im Freien,
legte sich hinter eine Hecke, schlief ein, erwachte jedoch bald wieder
und irrte von neuem umher, bald eilend, bald zögernd, rastlos selbst,
wenn er bisweilen rastete. In Hendon gedachte er irgendwo einzukehren,
allein sogar die Kinder vor den Türen schienen ihn argwöhnisch
anzublicken, der Mut fehlte ihm, einen Trunk oder einen Bissen Brot
zu fordern, und er suchte das Freie wieder auf, obwohl ihn die
vielstündige Wanderung, die ihn immer und immer wieder auf denselben
Fleck zurückführte, fast gänzlich erschöpft hatte.

Um neun Uhr abends wagte er sich endlich in ein kleines Gasthaus in
Hatfield hinein. Im Schenkstübchen am Feuer saßen einige ländliche
Arbeiter. Sie machten Platz für den unbekannten Gast, allein er setzte
sich in den fernsten Winkel und aß und trank allein, seinem ermüdeten
Hund von Zeit zu Zeit ein Stück zuwerfend. Die Arbeiter unterhielten
sich von ganz gewöhnlichen Dingen, und er schlummerte schon ein, als
lärmend ein Mann eintrat, der halb Hausierer, halb Marktschreier zu
sein schien und sogleich anfing, seine Waren ruhmrederisch und unter
mannigfachen Scherzen, wie sie zu dem Orte sich schicken mochten,
anzupreisen.

«Diese Kügelchen hier», sagte er in Erwiderung auf eine Frage eines der
Arbeiter, «sind ein untrügliches und unfehlbares Mittel, aus allerlei
Art Zeug alle Arten von Flecken auszutilgen. Hat eine Dame ihre Ehre
befleckt, so braucht sie nur ein solches Kügelchen zu genießen. Will
ein Herr seine Ehre beweisen, kann er's ebensogut mit 'nem solchen
Kügelchen tun als mit 'ner Pistolenkugel und noch besser, denn der
Geschmack ist viel schlechter. Wer kauft? Das Stück 'nen Penny --
oder auch zwei Halbpence oder vier Heller -- mir ist's ganz gleich.
Sie gehen so reißend ab, daß sie nur selten zu haben sind; vierzehn
Wassermühlen, sechs Dampfmaschinen und eine galvanische Batterie sind
unaufhörlich in Arbeit und können nicht schnell genug fabrizieren, um
die Käufer zu befriedigen, obgleich die angestellten Arbeiter sich
totarbeiten und die Witwen mit zwanzig Pfund jährlich für jedes Kind
pensioniert werden und mit 'ner Prämie für Zwillinge. Alle Flecke gehn
davon aus, Fettflecke, Wein- und Farbe- und Wasser- und Blutflecke.
Schauen Sie hier! Da ist ein Fleck auf dem Hute 'nes Gentleman, den ich
'runterbringen werde, eh' er mir 'nen Krug Ale bringen lassen kann.»

«Wollt Ihr wohl meinen Hut liegen lassen!» rief Sikes emporschreckend.

«Sir,» fuhr der Hausierer, den Arbeitern zublinzelnd, fort, «ich werde
den Fleck 'runter haben, eh' Sie zu mir herkommen können. Gentlemen,
Sie bemerken den dunkeln Fleck auf dem Hute des Gentleman, nicht größer
als ein Schilling, aber dicker als eine halbe Krone. Gleichviel, ob's
ein Fettfleck ist, oder ein Wein-, ein Farbe-, ein Wasser- oder ein
Blutfleck --»

Er kam nicht weiter, denn Sikes stieß mit einer schrecklichen
Verwünschung den Tisch um, entriß ihm den Hut, schritt wütend
aus dem Hause hinaus und wandte sich in derselben Verwirrung und
Unentschlossenheit, die ihn, ihm selber zum Trotze, den ganzen Tag
nicht hatte verlassen wollen, wieder nach der Stadt zurück. Vor dem
Posthause stand eine Londoner Diligence, und sorgfältig den Schein
ihrer Laternen meidend, näherte er sich ahnungsvoll, um zu horchen.

Er hatte eine Zeitlang dagestanden, als ein Wildwärter zu dem
Kondukteur trat, der am Fenster des Bureaus auf seine Abfertigung
wartete, und ihn fragte, ob es nichts Neues gäbe.

«Das Korn ist ein bissel gestiegen», lautete die Antwort. «Auch hörte
ich von 'ner Mordtat, begangen in der Gegend von Spitalsfield -- doch
wer weiß? Es wird entsetzlich gelogen.»

«Es ist vollkommen wahr», nahm ein Reisender das Wort. «Es ist eine
höchst schauderhafte Mordtat gewesen.»

«Ist sie denn an einem Manne oder an einer Frau begangen, Sir?»

«An einem Mädchen, und man sagte --»

Hier wurde der Kutscher ungeduldig und rief dem Kondukteur zu, daß er
sich beeilen möchte.

«Komme schon,» rief der Kondukteur heraustretend zurück, «wie auch die
reiche junge Dame schon kommt, die sich in mich verlieben wird, ich
weiß nur nicht, wann.»

Er stieg hinauf, stieß in sein Horn, und die Diligence rasselte fort.

Sikes stand da, anscheinend unbewegt und nur zweifelhaft, wohin er sich
wenden sollte. Endlich schlug er den Weg nach St. Albans ein.

Als er die Stadt hinter sich hatte und sich in der Finsternis auf der
einsamen Straße befand, bemächtigte sich seiner eine Beängstigung,
so furchtbar, wie wenn sie ihm das Herz abdrücken wollte. Alles um
ihn her, wirkliche Gegenstände wie Schatten, ob es sich regen mochte
oder nicht, nahm eine schreckliche Gestalt an; allein noch unendlich
fürchterlicher war die greuliche der Erschlagenen, die ihm dicht auf
den Fersen mit feierlichen, geisterhaften Schritten nachfolgte. Er sah
sie deutlich in der Finsternis, hörte ihre Kleider in den Blättern
rauschen, und jeder Windhauch führte seinem Ohre ihr letztes leises
Ächzen zu. Stand er still, so tat sie es ihm nach; lief er, so folgte
sie ihm auch -- nicht im Laufe, was ihm eine Herzenserleichterung
gewesen sein würde -- sondern wie eine Leiche, begabt nur mit
mechanischer Bewegungskraft und getragen von einem traurigen, langsam
daherrauschenden und sich weder verstärkenden noch abnehmenden
Lufthauche.

Mehreremal drehte er sich mit einem verzweifelten Entschlusse um,
gewillt, das Phantom zu verscheuchen, und wenn es ihn mit seinen
Blicken tötete; aber dann stand ihm das Haar zu Berge und das
Blut still, denn die Gestalt hatte sich mit ihm umgedreht und war
fortwährend hinter ihm. Am Morgen war sie vor ihm hergegangen --
jetzt folgte sie ihm. Er stellte sich mit dem Rücken an die Wand
eines steilen Grabens und hatte das Gefühl, daß sie, in deutlichen
Umrissen gegen den kalten Nachthimmel abstechend, vor ihm stand.
Er warf sich nieder auf die Straße, und sie stand ihm zu Häupten,
aufgerichtet, stumm und regungslos, gleich einem lebendigen Grabsteine
mit blutgeschriebener Inschrift.

Sage niemand, daß Mörder der Gerechtigkeit entgingen oder daß die
Vorsehung schlummere! Der Mörder Sikes erduldete in einer einzigen
Minute die Angst und Pein eines gewaltsamen Todes hundertfach.

Er erblickte einen Schuppen in einem Felde, welcher ein Obdach für die
Nacht darbot. Vor der Türe desselben standen drei hohe Pappelbäume, die
das Innere noch finsterer machten, und der Wind säuselte unheimlich in
ihren Blättern. Es war unmöglich, er konnte nicht bis zu Tagesanbruch
fortwandern und streckte sich dicht an der Wand nieder -- um neuen
Qualen zum Raube zu werden. Denn jetzt trat ein Gesicht vor ihn, noch
schrecklich beharrlicher und grausiger als das, welchem er entronnen
war. Zwei starre, halbgeöffnete Augen, glanzlos und gläsern, erschienen
ihm mitten in der Finsternis, hatten ihr eignes Licht, gaben aber keins
von sich. Es waren ihrer nur zwei, aber sie waren überall. Bedeckte
er seine Augen, so stand sein Zimmer mit allem, was es enthielt, so
deutlich vor ihm, wie wenn er sich darin befände. Alles war an seinem
Orte, auch die Leiche an dem ihrigen, und ihre Augen waren so gläsern
und starr wie in der Minute, als er hinausschlich. Er sprang auf und
eilte wieder in das Freie. Die Gestalt war hinter ihm. Er ging in den
Schuppen zurück und drückte sich wieder dicht an die Wand, und die
Augen waren wieder da, noch bevor er sich niedergelegt hatte.

Er bebte an allen Gliedern, und kalter Angstschweiß bedeckte ihn von
Kopf bis zu Füßen, als plötzlich aus weiter Ferne verwirrtes Rufen und
Schreien an sein Ohr drang. Es erschien ihm als eine Wohltat, eine
wirkliche Ursache zu Furcht und Schrecken zu erhalten. Kraft und Mut
kehrten ihm bei der Aussicht auf persönliche Gefahr zurück, er raffte
sich auf, eilte hinaus und sah den Himmel weithin von einer furchtbaren
Feuersbrunst gerötet. Die Sturmglocke ertönte, und lauter und lauter
wurde der Lärm und das Getöse. Es war ihm, als wäre ein neues Leben
in ihm erwacht. Er rannte, sein Hund wie toll vor ihm her, nach der
Richtung hin, über Hecken und Gräben, Mauern und Tore, kein Hindernis
achtend, und langte atemlos an.

Es standen viele Häuser in Flammen, und die Angst, das Geräusch, die
Verwirrung waren grenzenlos. Er schrie selbst mit, bis er heiser war,
und stürzte sich, um seinem Gedächtnisse und sich selbst zu entfliehen,
in den dichtesten Haufen, arbeitete bald an den Spritzen, erstieg bald
auf wankenden Leitern die höchsten Dachgiebel, war überall und trotzte
jeder Gefahr; allein er schien ein bezaubertes Leben zu haben und
empfand, bis der Tag graute, keine Spur von Ermüdung, trug nicht die
kleinste Brandwunde, keine Beule, keine Schramme davon.

Als jedoch die wahnsinnige Aufregung vorüber war, kehrte ihm mit
zehnfacher Gewalt das schreckliche Bewußtsein seines Verbrechens
zurück. Er blickte argwöhnisch umher, denn die Leute standen hier
und da beieinander und redeten untereinander, und er fürchtete, der
Gegenstand ihrer Gespräche zu sein. Der Hund gehorchte seinem Winke,
und beide stahlen sich davon. Die Wärter einer Spritze forderten ihn
auf, ihren Morgenimbiß mit ihnen zu teilen. Er nahm ein Stück Brot
und einen Trunk Bier an. Sie waren aus London und fingen an, von der
Mordtat zu sprechen. «Er ist nach Birmingham gegangen,» sagte einer
von ihnen, «aber sie werden ihn bald fassen, denn die Polizei hat ihre
Späher schon ausgeschickt, und bis morgen wird nur der eine Schrei im
ganzen Lande sein. >Wo ist der Mörder?<»

Er eilte fort und ging, solange die Füße ihn tragen wollten, warf
sich an einem entlegenen Orte nieder, verfiel in einen langen, aber
unruhigen, oft unterbrochenen Schlaf, setzte unentschlossen und
ungewiß, geängstet von der Furcht vor einer zweiten einsamen Nacht,
seine Wanderung wieder fort und faßte plötzlich den verzweifelten
Entschluß, nach London zurückzukehren.

«Kann doch wenigstens dort mit jemand sprechen,» dachte er, «und hab'
ein gutes Versteck. Sie suchen mich da am letzten. Ich halte mich
'ne Woche still, zwinge Fagin, zu blechen und schiffe 'nüber nach
Frankreich. Gott verdamm mich, ich wag's!»

Sein Plan war, nach Dunkelwerden auf Schleichwegen Fagins Wohnung zu
erreichen. Aber der mit ihm vermißte Hund konnte seine Entdeckung
und Verhaftung veranlassen. Er beschloß, ihn zu ersäufen, hob einen
schweren Stein auf und knüpfte denselben in sein Taschentuch. Ein
Gewässer war in der Nähe, er lockte den Hund, allein lange mit
vergeblicher Mühe; die Blicke seines brutalen Herrn mochten den
Instinkt des Tieres noch verschärft haben. Sikes schmeichelte und
drohte, der Hund kroch endlich zu ihm heran, sprang aber, als er sich
plötzlich gefaßt fühlte, zurück, lief davon, und Sikes mußte seine
Wanderung allein fortsetzen.




49. Kapitel.

    Die endlich stattfindende Unterredung zwischen Monks und Mr.
    Brownlow.


Es wurde dunkel, als Mr. Brownlow mit zwei Männern aus einem Mietswagen
stieg, der vor seinem Hause hielt; die letzteren halfen einem dritten
Manne heraus und drängten ihn rasch durch die geöffnete Tür hinein. Der
Mann war Monks.

Brownlow ging ihnen schweigend in ein hinteres Zimmer voran. Vor
der Tür desselben stand Monks widerstrebend still, und die beiden
handfesten Männer sahen Brownlow fragend an.

«Entweder oder», sagte Brownlow. «Die Folgen des einen wie des andern
sind ihm bekannt. Weigert er sich, hineinzugehen, so führt ihn aus dem
Hause, ruft die Polizei zu Hilfe und klagt ihn in meinem Namen als
Kapitalverbrecher an.»

«Wie können Sie sich unterstehen, mich so zu nennen?» fuhr Monks auf.

«Wie können Sie es wagen, mich zu einer Anklage gegen Sie zu drängen,
junger Mensch?» entgegnete Brownlow, ihn sehr bestimmt anblickend.
«Werden Sie unsinnig genug sein, mein Haus zu verlassen? Laßt ihn los!
So, Sir. Sie können jetzt gehen -- und wir können Ihnen nachfolgen.
Aber ich gebe Ihnen mein Wort darauf, sobald Sie den Fuß vor die Tür
setzen, sind Sie auch schon wegen Betruges und Raubes verhaftet. Ich
bin fest entschlossen. Sind Sie es auch -- nun wohl! -- aber Ihr Blut
kommt auf Ihr eigenes Haupt.»

«Aus wessen Macht bin ich auf offener Straße aufgegriffen und von
diesen Schuften hierher gebracht worden?»

«Ich verantworte, was die Leute getan haben. Beklagen Sie sich über
Freiheitsberaubung -- es stand in Ihrer Gewalt, ihr auf dem Wege
hierher ein Ende zu machen; Sie erachteten es aber selbst für rätlich,
sich ruhig zu verhalten. Wollen Sie die Gesetze anrufen -- tun Sie's;
allein ich werde es gleichfalls tun und Ihnen keine Milde mehr zeigen,
mich nicht bemühen, Sie zu retten, wenn die Sachen erst einmal vor den
Richter gekommen sind.»

Monks war offenbar unentschlossen geworden.

«Entschließen Sie sich rasch», fuhr Brownlow mit ruhiger Festigkeit
fort. «Wollen Sie, daß ich Anklagen gegen Sie vorbringe, deren Ausgang
ich schaudernd vorhersehe, so wissen Sie, was Sie zu tun haben;
wünschen Sie Nachsicht und Vergebung von mir und den von Ihnen schwer
Gekränkten, so treten Sie hinein und nehmen Sie, ohne ein Wort zu
sagen, dort auf jenem Stuhle Platz, der schon zwei Tage auf Sie
gewartet hat.»

Monks zögerte noch einige Augenblicke, ging indes endlich hinein und
setzte sich. Brownlow befahl den beiden Männern, die Tür zu verriegeln
und wieder zur Stelle zu sein, wenn er klingelte.

«Eine saubere Behandlung, die ich von dem ältesten Freunde meines
Vaters erfahre», sagte Monks, Hut und Mantel ablegend.

«Gerade weil ich Ihres Vaters ältester Freund bin, junger Mann,»
erwiderte Brownlow, «weil einst die Hoffnungen und Wünsche meiner
glücklichen Jugendzeit an ihn sich anknüpften und an ein holdes Wesen
von seinem Blute, das in seinen jungen Jahren zu Gott zurückkehrte und
mich einsam und allein hier zurückließ; -- weil er, noch ein Knabe,
mit mir an seiner einzigen Schwester Sterbebette kniete, an dem Morgen
kniete, der sie, wenn es der Himmel nicht anders gewollt hätte, zu
meinem Weibe gemacht haben würde: -- weil mein wundes Herz von der
Zeit an bis zu seinem Tode bei all seinen Prüfungen und Irrtümern an
ihm hing; -- weil alle teure Erinnerungen an ihn mein Herz erfüllen
und selbst durch Ihren Anblick erneuert werden; -- das alles ist es,
weshalb ich Sie jetzt nachsichtig behandle -- ja, Eduard Leeford, sogar
jetzt -- Sie, der Sie erröten müssen, dieses Namens so unwürdig zu
sein.»

«Was hat der mit der Sache zu schaffen?» fragte Monks, der verstockt
und stumm-verwundert die Bewegung des alten Herrn gewahrt hatte. «Was
gibt mir der Name?»

«Freilich gilt er Ihnen nichts», versetzte Brownlow. «Er war aber
der Ihrige, und noch glüht und erhebt mir altem Manne in so weiter
Zeitenferne das Herz wie sonst, wenn ich ihn nur von fremden Lippen
nennen höre. Ich bin sehr, sehr erfreut, daß Sie ihn mit einem anderen
vertauscht haben.»

«Das klingt alles gar prächtig», sagte Monks nach einem langen
Stillschweigen, während dessen er sich trotzig hin und her gewiegt und
Brownlow, die Augen mit der Hand bedeckend, dagesessen hatte. «Aber was
wollen Sie von mir?»

«Sie haben einen Bruder, dessen Name, in Ihr Ohr geflüstert, als ich
auf der Straße hinter Ihnen ging, fast allein schon genügte, Sie zu
veranlassen, erstaunt und erschreckt mich hierher zu begleiten.»

«Ich habe keinen Bruder. Sie wissen, daß ich das einzige Kind war,
wissen es ebensogut wie ich selbst.»

«Hören Sie, was ich weiß, und Sie werden schon anders reden, schon
aufmerken auf das, was ich Ihnen sage. Ich weiß allerdings, daß Sie der
einzige und höchst unnatürliche Sprößling aus der unseligen Verbindung
waren, zu welcher elender Familienstolz Ihren unglücklichen Vater fast
noch als Knaben nötigte.»

«Es gilt mir gleich, wie harte Ausdrücke Sie wählen mögen», unterbrach
ihn Monks mit einem höhnischen Lachen. «Sie sind mit der Sache bekannt,
und das ist mir genug.»

«Mir sind aber auch das Elend und die langen Qualen bekannt,» fuhr
Brownlow fort, «welche der unpassenden Verbindung folgten. Ich weiß,
unter welcher Pein das unglückliche Paar seine Kette durch die ihm
vergällte Welt nachschleppte; weiß, daß auf förmliche Gleichgültigkeit
Beleidigungen, Widerwille, Haß und Abscheu folgten, bis sie sich
endlich trennten, um fern voneinander zu leben und in anderen Kreisen
die lange Quälerei zu vergessen. Und Ihre Mutter vergaß sie bald, Ihren
Vater aber drückte sie noch jahrelang zu Boden.»

«Was weiter, als sie sich voneinander getrennt hatten?»

«Die zehn Jahre ältere Gattin vergaß unter Zerstreuungen auf dem
Festlande den jugendlichen Gatten, der daheim sein geknicktes Leben
vertrauerte, bis er eine Verbindung mit neuen Freunden anknüpfte -- und
zum wenigsten dieser Umstand ist zu Ihrer Kenntnis gelangt.»

«Nein», sagte Monks, die Blicke wegwendend und auf den Boden stampfend,
wie jemand, der alles abzuleugnen entschlossen ist. «Nein!»

«Ihr Benehmen und Ihre Handlungen überzeugen mich, daß Sie ihn nie
vergessen, nie aufgehört haben, mit Bitterkeit daran zurückzudenken.
Ich rede von der Zeit vor fünfzehn Jahren, wo Sie erst elf Jahre alt
waren, und Ihr Vater nur einunddreißig zählte, -- denn, wie gesagt, er
war fast noch ein Knabe, als ihn sein Vater zu heiraten zwang. Muß ich
Dinge erwähnen, die einen Schatten auf das Andenken Ihres Erzeugers
werfen, oder wollen Sie es mir ersparen und mir die Wahrheit enthüllen?»

«Ich habe nichts zu enthüllen!» erwiderte Monks in offenbarer
Verwirrung. «Reden Sie weiter, wenn Sie es nicht lassen können.»

«Nun wohl», sagte Brownlow. «Die neuen Freunde waren zunächst ein
Flottenoffizier, der sich aus dem aktiven Dienste zurückgezogen, und
dessen Frau vor einem halben Jahre gestorben war. Sie hatten mehrere
Kinder gehabt, von denen nur zwei die Mutter überlebten, beide Töchter,
die eine schön und neunzehn, die andere ein Kind, zwei bis drei Jahre
alt.»

«Was geht das mich an?» fragte Monks.

«Sie wohnten», fuhr Brownlow, anscheinend ohne die Unterbrechung zu
beachten, fort, «in einer Grafschaft, in welche Ihren Vater seine
Streifereien geführt, und wo auch er seinen Wohnsitz aufgeschlagen
hatte. Bekanntschaft, vertraulicher Umgang und Freundschaft folgten
schnell aufeinander. Ihr Vater war begabt, wie es wenige Männer sind
-- er hatte seiner Schwester Züge und Seele. In dem Maße, wie der alte
Offizier ihn kennen lernte, begann er, ihn wahrhaft zu lieben. Daß es
dabei geblieben wäre! Doch seine Tochter tat dasselbe.»

Der alte Herr hielt inne, sprach aber bald weiter, da er sah, daß sich
Monks in die Lippen biß und die Blicke an den Boden heftete.

«Sie war am Schlusse des Jahres mit Ihrem Vater verlobt, feierlich
verlobt, Ihr Vater der Gegenstand der ersten, treuinnigen, glühenden,
einzigen Liebe eines arglosen, unerfahrenen Mädchens.»

«Ihre Erzählung wird lang», bemerkte Monks, unruhig hin und her rückend.

«Sie ist eine wahre und traurige,» versetzte Brownlow, «und Geschichten
dieser Art pflegen lang zu sein; wenn sie von ungetrübtem Glücke
handelte, so wäre sie ohne Zweifel sehr kurz. -- Endlich starb einer
der reichen Verwandten, dessen Einfluß zu verstärken Ihr Vater
von Ihrem Großvater geopfert worden war, und hinterließ ihm seine
Panazee für alles Wehe -- Geld. Er mußte nach Rom eilen, wo der
Erblasser gestorben war und seine Angelegenheiten in großer Verwirrung
hinterlassen hatte, erkrankte selbst am Tage nach seiner Ankunft und
starb nach einiger Zeit, ohne für eine letztwillige Verfügung Sorge
getragen zu haben, so daß sein ganzes Vermögen Ihrer Mutter und Ihnen
zufiel, die mit Ihnen nach Rom eilte, sobald sie in Paris die Kunde
seines Todes erhalten.»

Monks hielt hier den Atem an und hörte Brownlow in großer Spannung zu,
obgleich er die Blicke nicht nach ihm hinwandte. Als Brownlow schwieg,
veränderte er seine Stellung, wie wenn er sich plötzlich erleichtert
fühlte, und fuhr mit dem Tuch über sein glühendes Antlitz. Brownlow
sprach langsam und die Blicke fest auf ihn heftend, weiter: «Bevor er
außer Landes ging und als er London berührte, kam er zu mir.»

«Davon habe ich nie gehört», unterbrach Monks in einem Tone,
der Unglauben ausdrücken sollte, doch mehr auf eine unangenehme
Überraschung hindeutete.

«Er kam zu mir und ließ unter anderen Gegenständen ein von ihm selbst
gemaltes Bild des unglücklichen Mädchens, seiner Verlobten, zurück,
das er in anderen Händen nicht zu lassen wünschte, auf seiner eiligen
Reise aber nicht wohl mitnehmen konnte. Er war fast zu einem Schatten
zusammengeschwunden, sprach verstört von begangenem Ehrenraube und
Verderben, das er angerichtet, und kündigte mir an, daß er entschlossen
wäre, sein ganzes Vermögen, ob auch mit großem Verluste, in bares
Geld umzuwandeln, Ihrer Mutter und Ihnen einen Teil seiner neu zu
erwerbenden Erbschaft auszusetzen, und das Land zu verlassen -- ich
wußte nur zu wohl, daß er nicht allein gehen würde --, um es nie
wiederzusehen. Mehr bekannte er sogar mir, seinem alten Jugendfreunde,
nicht, dessen starke Zuneigung in der Erde wurzelte, die sie bedeckte,
die beiden so teuer gewesen war. Er versprach mir, zu schreiben und
mir alles zu sagen, und mich dann noch ein -- das letztemal hienieden,
wiederzusehen. Ach, es war schon das letztemal. Ich bekam keinen Brief
und sah ihn nicht wieder. Als ich vernahm, daß er tot war, begab
ich mich nach dem Schauplatz seiner -- wie die Welt es nennen würde
-- sündlichen Liebe, um, wenn ich meine Befürchtungen wahr geworden
fände, dem verirrten Mädchen ein mitleidiges Herz und eine Zuflucht
anzubieten. Allein die Familie hatte vor einer kurzen Zeit die
Angelegenheiten geordnet und war bei Nacht abgereist, niemand wußte
wohin.»

Monks atmete freier und blickte mit einem triumphierenden Lächeln
umher. Brownlow rückte seinen Stuhl näher zu ihm und sagte: «Als Ihr
Bruder -- ein verlorener, schwacher, in Lumpen gehüllter Knabe -- nicht
durch Zufall, sondern durch eine höhere Fügung in meinen Weg geworfen
und von mir gerettet wurde --»

«Wie?!» rief Monks in heftiger Spannung aus.

«Von mir», wiederholte Brownlow. «Ich sagte es Ihnen, daß Sie schon
aufmerken würden auf das, was ich Ihnen zu sagen gedächte. Ja, von
mir -- ich sehe, Ihr schlauer Verbündeter hat Ihnen meinen Namen
verschwiegen, obwohl er nicht annehmen konnte, daß Ihnen derselbe
bekannt wäre. Während sich Ihr Bruder als Kranker und Wiedergenesener
in meinem Hause befand, erkannte ich zu meinem lebhaften Erstaunen
seine große Ähnlichkeit mit dem erwähnten Bilde. Schon im ersten
Augenblicke, als ich ihn sah, erinnerten mich seine Züge an einen
alten Freund, nur daß mir alles unbestimmt blieb, wie wenn ich mir
Traumbilder vergeblich deutlich und klar vor die Seele zurückzurufen
suchte. Ich brauche Ihnen nicht zu sagen, daß er mir entführt wurde,
ehe ich seine Geschichte kennen lernte --»

«Warum nicht?» fragte Monks hastig.

«Sie wissen es sehr gut.»

«Ich?»

«Sie leugnen vergeblich und sollen bald sehen, daß ich noch mehr weiß.»

«Sie -- Sie können nichts gegen mich beweisen», stotterte Monks. «Tun
Sie's, wenn Sie's imstande sind.»

«Wir werden sehen», sagte der alte Herr mit einem durchdringenden
Blicke. «Der Knabe wurde mir entführt, und meine Bemühungen, ihn
wieder aufzufinden, waren vergeblich. Da Ihre Mutter tot war, so
konnten Sie allein, wenn irgend jemand, das Geheimnis enthüllen, und
da Sie, wie ich gehört hatte, auf Ihrer Pflanzung in Westindien sich
aufhielten -- wohin Sie nach Ihrer Mutter Tode gegangen waren, um den
Folgen Ihres ruchlosen Lebenswandels hier in England zu entgehen --,
so reiste ich Ihnen nach. Sie hatten sich unterdes wieder entfernt,
und man glaubte, daß Sie sich in London befänden, doch vermochte
niemand genauere Nachweisungen zu geben. Ich kehrte zurück. Ihren
Geschäftsführern war Ihr Wohnort vollkommen unbekannt; sie sagten,
daß Sie ebenso geheimnisvoll kämen und gingen, wie Sie es immer getan
hätten, bisweilen wochen- und monatelang nicht erschienen und aller
Wahrscheinlichkeit nach mit den schandbaren Menschen sich umhertrieben,
denen Sie sich zugesellt, seit Sie ein trotziger, unlenksamer Knabe
waren. Ich hörte nicht auf, sie zu befragen, in Tätigkeit zu erhalten.
Ich durchwanderte die Straßen bei Nacht wie bei Tage, allein meine Mühe
war bis vor zwei Stunden fruchtlos, wo ich Ihrer endlich zum ersten
Male ansichtig wurde.»

«Und da Sie mich nun aufgefunden haben,» nahm Monks, sich dreist
erhebend, das Wort, «was mehr? Betrug und Raub sind volltönende Worte
-- und gerechtfertigt, wie Ihnen scheint, durch die eingebildete
Ähnlichkeit eines kleinen Landstreichers mit der Pinselei eines längst
Verstorbenen. Allein Sie wissen nicht einmal, ob aus der Verbindung des
letzteren mit dem erwähnten Mädchen ein Kind entsproß -- wissen das
nicht einmal!»

«Ich weiß es wirklich erst seit vierzehn Tagen», erwiderte Brownlow,
gleichfalls aufstehend. «Sie haben einen Bruder, wissen es und kennen
ihn. Es war ein Testament vorhanden, das Ihre Mutter vernichtete, die
Ihnen bei ihrem Tode das Geheimnis und den sündigen Gewinn hinterließ.
Das Testament nahm Bezug auf ein Kind, das Ihrem Vater noch geboren
werden möchte; es wurde geboren, Sie trafen mit ihm zusammen, und
seine Ähnlichkeit mit Ihrem Vater erweckte böse Ahnungen in Ihnen. Sie
suchten seinen Geburtsort auf, wo Beweise, lange unterdrückte Beweise
seiner Geburt und Herkunft vorhanden waren. Sie vernichteten sie, und,
wie Sie Ihrem jüdischen Schand- und Schuldgenossen sagten, sie liegen
jetzt auf dem Grunde des Stromes, und die alte Hexe, die sie seiner
Mutter nahm, fault in ihrem Sarge. Unwürdiger Sohn, Lügner, Feigling,
der du nachts mit Räubern und Mördern in finsteren Gemächern verkehrst,
-- der du durch schändliche List an dem kläglichen Tode einer
Unglücklichen schuld bist, deren Wert Millionen von deinesgleichen
aufwog, -- der du von deiner Wiege an dem Herzen deines Vaters nur
Bitterkeit und Galle warst, -- du, in dessen angefaultem Innern die
schlechtesten Leidenschaften so lange eiterten, bis sie einen Ausbruch
in der scheußlichen Krankheit fanden, die dein Antlitz zu einem Spiegel
deiner teuflischen Seele gemacht hat, -- Eduard Leeford, setzt du mir
auch jetzt noch Trotz entgegen?»

«Nein, nein, nein!» stöhnte der durch so gehäufte Beschuldigungen
überwältigte Feigling.

«Jedes Wort,» rief Brownlow aus, «jedes Wort, das zwischen dir und dem
über alles schändlichen Bösewicht gewechselt worden, ist mir bekannt.
Schatten an der Wand haben dein Geflüster vernommen und meinem Ohr
zugeführt; der Anblick des unschuldigen, verfolgten Kindes hat selbst
das Laster ergriffen und ihm den Mut und fast die Wesenheit der Tugend
verliehen. Ein Mord ist begangen, an welchem du mindestens moralisch
teilgenommen hast!»

«Nein, nein», unterbrach Monks. «Ich -- ich weiß nichts davon. Ich ging
eben, um zu erfahren, was Wahres an der Sache wäre, als Sie mich mit
sich fortführten. Die Veranlassung der Tat war mir unbekannt -- ich
glaubte, sie wäre nur durch einen gewöhnlichen Streit gegeben worden.»

«Sie war keine andere als die teilweise Enthüllung Ihrer Geheimnisse»,
sagte Brownlow. «Wollen Sie dieselben jetzt ganz offenbaren?»

«Ja, ich will's!»

«Alles vor Zeugen wiederholen und eine wahrhafte Aufzeichnung durch
Ihre Namensunterschrift beglaubigen?»

«Auch das verspreche ich.»

«Ruhig in meiner Wohnung verweilen, bis es geschehen ist, und sich
mit mir an den Ort begeben, den ich für den geeignetsten halte, dem
Dokumente die vollkommenste Gültigkeit zu verschaffen?»

«Wenn Sie darauf bestehen, will ich auch das tun.»

«Sie müssen noch mehr tun, dem guten, unschuldigen Kinde Ersatz
leisten. Sie haben die Verfügungen Ihres väterlichen Testaments nicht
vergessen. Bringen Sie dieselben, soweit sie Ihren Bruder betreffen,
zur Ausführung und gehen Sie dann, wohin es Ihnen beliebt. Sie dürfen
einander in dieser Welt nicht mehr begegnen.»

Während Monks auf und ab ging und der Aufforderung Brownlows und
listigen Ausflüchten, zwischen Furcht und Haß schwankend, nachsann,
wurde plötzlich die Tür aufgeschlossen und geöffnet, und herein trat in
heftiger Aufregung Mr. Losberne.

«Er wird ergriffen, wird heute abend ergriffen werden!» rief er.

«Der Mörder?» fragte Brownlow.

«Ja, ja. Sein Hund hat die Polizei auf die Spur geführt. Sein
Schlupfwinkel ist von allen Seiten umstellt, und die Behörden haben
hundert Pfund ausgesetzt.»

«Ich lege noch fünfzig zu und will es auf der Stelle mit meinen eigenen
Lippen verkünden. Wo ist Maylie?»

«Harry -- er warf sich, sobald er Sie mit dem jungen Menschen im
Mietswagen sah, zu Pferde und sprengte fort, um sich den Verfolgern des
Mörders anzuschließen.»

«Hörten Sie nichts von dem Juden?»

«Er wird in diesem Augenblick bereits festgenommen sein.»

«Haben Sie Ihren Entschluß gefaßt?» fragte Brownlow Monks leise.

«Ja. Sie -- Sie werden mich nicht ausliefern?»

«Nein. Aber Sie bleiben hier, bis ich zurückkehre. Ihre Sicherheit
hängt einzig davon ab.»

Brownlow und Losberne entfernten sich, und die Tür wurde wieder
verschlossen.

«Was haben Sie ausgerichtet?» fragte Losberne flüsternd.

«Soviel ich hoffen konnte und mehr. Veranstalten Sie auf übermorgen
abend die Zusammenkunft. Wir werden ein paar Stunden früher da sein,
aber Ruhe bedürfen, besonders die junge Dame, die vielleicht größerer
Festigkeit benötigt sein möchte, als Sie und ich jetzt voraussehen
können. Doch mir kocht das Blut in den Adern, das arme ermordete
Geschöpf zu rächen. Wohin muß ich meine Schritte richten?»

«Eilen Sie nur zuvörderst nach dem Polizeiamte; ich will hierbleiben.»

Die Herren nahmen hastigen Abschied voneinander, beide in einem
unbezähmbaren Fieber von Aufregung.




50. Kapitel.

    Verfolgung und Entkommen.


Unweit der Stelle des Themseufers, wo die Kirche von Rotherhithe steht
und die Gebäude am erbärmlichsten und die Fahrzeuge auf dem Strome
am schwärzesten aussehen vom Kohlenstaube und dem Rauche der eng
aneinander gebauten niedrigen Häuser, befindet sich heutigestags die
schmutzigste, widerwärtigste und unheimlichste der vielen in London
versteckten und der großen Mehrzahl der Bewohner der Hauptstadt selbst
dem Namen nach unbekannten Örtlichkeiten.

Um zu ihr zu gelangen, muß man sich durch ein Labyrinth von kotigen
und engen Straßen hindurchwinden, die von den rohesten und ärmsten
Uferbewohnern erfüllt und ihrem Verkehre gewidmet sind. In den Läden
schaut man die wohlfeilsten und uneinladendsten Nahrungsmittel, an
den Fenstern und Türen der Altkleiderhändler die verschiedenartigsten
Lumpen. Man arbeitet und stößt sich mühsam weiter durch das
Gedränge unbeschäftigter Menschen der niedrigsten Klasse, Last- und
Kohlenträger, frecher Weiber, zerlumpter Kinder, des recht eigentlichen
Themseabschaums, indem ekelhafte Gegenstände und Düfte in allen
Richtungen das Auge und den Geruchsinn beleidigen und das Ohr durch
verwirrtes Geräusch aller Art betäubt wird. Gelangt man endlich in die
noch entlegneren, minder besuchten Winkelgassen, so scheinen wankende
Häuser zu beiden Seiten mit augenscheinlichem Einsturze zu drohen,
und man sieht, wohin man blickt, halb eingefallene Schornsteine,
erblindete oder zerschlagene Fenster und was nur sonst an Armut und
Vernachlässigung erinnern mag.

In einer solchen Umgebung, jenseits Dockhead im Borough Southwark,
befindet sich die Jakobsinsel, umgeben von einem Sumpfgraben von sechs
bis acht Fuß Tiefe und fünfzehn bis zwanzig Fuß Breite zur Flutzeit,
vormals der Mühlgraben genannt, jetzt bekannt unter dem Namen Folly
Ditch. Sie ist eine Art Strombucht und kann bei hohem Wasser durch
Öffnung der Schleusen bei den Leadmühlen, von welchen sie ihre alte
Benennung hat, ganz unter Wasser gesetzt werden. Steht man, wenn dies
geschieht, auf einer der hölzernen Brücken, die bei der Mühlengasse
über sie hinüberführen, so kann man sehen, wie die Bewohner der Häuser
zu beiden Seiten an den Hintertüren und Fenstern Eimer und Küchengerät
aller Art herunterlassen, um Wasser zu schöpfen, und erblickt hölzerne
Galerien, welche ein halbes Dutzend Hinterhäuser verbinden, mit
Löchern, aus denen sich auf die Lache hinunterschauen läßt; verklebte
und verstopfte Fenster, aus welchen Stangen hervorstehen zum Trocknen
von Wäsche, die nicht vorhanden ist; die denkbar engsten, dumpfigsten,
finstersten Gemächer; halbversunkene, mißfarbige Wände und zahllose
ähnliche Anzeichen des Verfalls und Elends.

Die Warenhäuser der Jakobsinsel stehen leer und haben weder Dächer
noch Fenster noch Türen. Dem lebhaften Verkehre, der hier vor einigen
Jahrzehnten stattfand, ist Verödung gefolgt. Die Häuser haben keine
Eigentümer, stehen unbewohnt oder werden erbrochen und bewohnt von
Leuten, die den Mut dazu und sonst keine Wohnstätte haben, bei welchen
entweder starke Beweggründe obwalten, in tiefer Verborgenheit zu
leben, oder die sich in der allerbedürftigsten und jammervollsten Lage
befinden.

In einem oberen Gemache eines dieser Häuser, das etwas abgesondert
stand, in anderen Beziehungen zu den verfallensten gehörte, aber stark
befestigte Türen und Fenster hatte, von welchen die hinteren auf das
beschriebene sumpfige Gewässer hinausgingen, saßen drei Männer in
tiefem düsteren Stillschweigen, einander von Zeit zu Zeit Blicke der
Bangigkeit und angstvollen Erwartung zuwerfend. Es waren Toby Crackit,
Tom Chitling und ein Raubgeselle von etwa fünfzig Jahren, dem einst die
Nase fast plattgeschlagen worden, und dessen Gesicht eine grauenvolle
Narbe hatte, ohne Zweifel gleichfalls infolge einer Schlägerei. Er war
ein zurückgekehrter Deportierter und hieß Kags.

«Ich wollte,» nahm endlich Toby, zu Chitling sich wendend, das Wort,
«daß Ihr Euch ein anderes Bayes ausgesucht hättet, als Euch die beiden
alten zu warm wurden, und nicht hierher gekommen wäret.»

«Freilich», stimmte Kags bei; «warum tat'st das nicht, Dummkopf?»

«Ich glaubte, Ihr würdet etwas vergnügter gewesen sein, mich zu sehen»,
antwortete Chitling mit trübseliger Miene.

«Ja seht, junger Herr,» sagte Toby, «wenn sich einer so exklusiv hält,
wie ich's getan habe, und somit in 'nem gemütlichen Hause sitzt,
da niemand reinguckt und das niemand umschnüffelt, so ist's ein
verfluchtes Ding, die Ehre 'nes Besuchs von 'nem jungen Gentleman in
Eurer Lage zu haben, so respektabel und angenehm es sonst sein mag,
nach Umständen Karten mit ihm zu spielen.»

«Besonders,» fügte Kags hinzu, «wenn der exklusive junge Gentleman
'nen Freund bei sich hat, der aus fremden Ländern eher zurückgekehrt
ist, als er erwartet wurde, und zu viel Bescheidenheit besitzt, um zu
wünschen, nach seiner Heimkehr den Richtern vorgestellt zu werden.»

Toby Crackit schwieg eine Zeitlang und fragte darauf Chitling, doch
nicht mehr in seinem leichtfertig renommistischen Tone, wann Fagin
ergriffen worden wäre.

«Heute nachmittag um zwei Uhr», erwiderte Tom. «Charley und ich
entkamen durch den Waschhausschornstein, und Bolter plumpste mit dem
Kopfe zuerst in 'ne leere Wassertonne hinein; aber seine langen Beine
standen heraus, und er wurde auch gefaßt.»

«Und Bet?»

«Sie ging, um die Leiche anzusehen, und fing an zu toben und zu rasen
bei dem Anblicke und wollte sich den Kopf einrennen. Sie legten ihr
drum 'ne Zwangsjacke an und brachten sie ins Tollhaus, wo sie noch ist.»

«Was ist denn aus dem Bates geworden?» fragte Kags.

«Er wird hier sein, sobald es dunkel geworden ist, und treibt sich
solange herum, wo er kann. Die aus 'n Krüppel sitzen alle, und die
ganze Schenkstube ist voll von Polizei; ich hab's mit meinen eigenen
Augen gesehen.»

«Da wird noch manch einer mit hineinverwickelt werden», bemerkte Toby,
sich auf die Lippen beißend.

«'s ist Schwurgerichtssaison,» sagte Kags, «und wenn Bolter gegen Fagin
aussagt, was er ohne Zweifel tun wird, so baumelt der Jude bei Gott
nach sechs Tagen.»

«Ihr hättet nur die Leute toben hören sollen», fuhr Chitling fort.
«Hätten die Schuker nicht wie Teufel gefochten, so wär ihnen Fagin vom
Volk entrissen. Er sah aus wie durch Kot und Blut gezogen, denn einmal
war er schon niedergeschlagen und hing sich an die Schuker, als wenn
sie seine teuersten Freunde gewesen wären. Sie mußten ihn in die Mitte
nehmen, und der andrängende wütende Haufen war wie 'ne Herde reißender,
nach seinem Blute lechzender Wölfe und lärmte wie besessen, und die
Weiber schrien, daß sie ihm das Herz aus'm Leibe reißen wollten.»

Alle drei saßen einige Minuten entsetzt und schweigend da, als
plötzlich auf der Treppe ein Geräusch ertönte und unmittelbar darauf
Sikes' Hund hereinsprang. Sie liefen an das Fenster; er mußte durch
irgendeine Öffnung hereingekommen sein; sein Herr war jedoch nicht zu
sehen.

«Was ist dies?» sagte Toby, nachdem sie vom Fenster zurückgetreten
waren. «Ich will doch hoffen, daß er nicht hierher kommt?»

«Wenn er das gewollt hätte, würd' er mit dem Hunde gekommen sein, der
gerade so aussieht, als wenn er weit hergelaufen wäre», meinte Kags.

«Aber woher kann er gekommen sein?» fuhr Toby fort. «Hm! er hat Fremde
in den andern Häusern gefunden, und hier ist er schon öfter gewesen.
Aber warum kommt er ohne ihn?»

«Er» (keiner nannte den Mörder bei seinem Namen), «er ist sicher übers
Wasser,» sagte Kags, «und er hat den Hund zurückgelassen, der sonst
nicht so ruhig daliegen würde.»

Als es dunkel geworden war, verschlossen sie den Fensterladen und
zündeten Licht an. Die schrecklichen Ereignisse der beiden letzten Tage
hatten sie mit Furcht und Entsetzen erfüllt. Sie schreckten bei jedem
Laute zusammen und flüsterten nur von Zeit zu Zeit ein paar Worte, als
wenn das Gespenst der Ermordeten im Hause umginge. Plötzlich wurde laut
an die Haustür geklopft. Crackit sah aus dem Fenster und erblaßte.
Sie berieten, und das Ergebnis war, daß er eingelassen werden müsse.
Crackit ging und kehrte bald darauf mit einem Manne zurück, der mehr
wie des Mörders fürchterlicher Geist als wie Sikes selber aussah, mit
seinen erdfahlen, eingefallenen Wangen, erloschenen, tiefliegenden
Augen und langgewachsenem Barte. Er wollte sich auf einen Stuhl am
Tische niederlassen, schauderte aber und schob den Stuhl dicht an die
Wand.

Kein Wort war noch gesprochen worden. Seine Blicke schweiften von
dem einen zum andern. Ward ein Auge aufgeschlagen und begegnete dem
seinigen, so wurde es augenblicklich wieder gesenkt. Als er endlich
das Stillschweigen brach, schreckten alle drei bei dem nie vernommenen
hohlen Tone seiner Stimme zusammen.

«Wie kam der Hund hier ins Haus?» fragte er.

«Allein. Vor drei Stunden.»

«Es heißt, daß Fagin eingezogen wäre. Ist's wahr oder gelogen?»

«Vollkommen wahr.»

Es trat ein abermaliges Schweigen ein.

«Geht alle zur Hölle!» hub Sikes endlich, mit der Hand über die Stirn
fahrend, wieder an. «Habt ihr mir nichts zu sagen?»

Es erfolgte eine unruhige Bewegung unter ihnen, aber niemand sprach.

«Ihr, der Ihr hier Herr im Hause spielt,» fuhr Sikes zu Crackit
gewandt, fort, «denkt Ihr mich zu verkaufen oder mich hier unterducken
zu lassen, bis die Hetze vorbei ist?»

«Ihr könnt bleiben, wenn Ihr Euch hier für sicher haltet», antwortete
Toby zögernd.

Sikes blickte oder machte vielmehr nur den Versuch, hinter sich an der
Wand hinaufzublicken, und sagte: «Ist -- ist sie -- ist die Leiche schon
beigesetzt?»

Das Kleeblatt schüttelte die Köpfe.

«Warum nicht?» fuhr er, ebenso hinter sich blickend, fort. «Warum
lassen sie ein so häßliches Ding über der Erde? -- Wer klopft da?»

Toby erwiderte, es wäre nichts zu fürchten, ging hinaus und trat mit
Charley Bates wieder herein. Sikes saß der Tür gegenüber, so daß die
Blicke des Knaben sogleich auf seine Gestalt fielen.

«Toby,» sagte Charley, «warum habt Ihr mir das unten nicht gesagt?»

Sikes sah die drei zusammenschrecken und hielt dem Knaben die Hand
zutunlich schmeichelnd entgegen, denn es bemächtigte sich seiner ein
unnennbares Entsetzen.

«Laß mich in ein anderes Zimmer gehen», sagte Charley, sich
zurückziehend.

«Charley», sagte Sikes, aufstehend und ein paar Schritte vortretend:
«wie -- kennst du mich nicht?»

«Kommt mir nicht näher!» rief der Knabe, noch weiter zurückweichend und
schaudernd dem Mörder in das Angesicht blickend. «Ihr Ungeheuer -- Ihr
Unmensch!»

Sikes stand auf halbem Wege still, und beide blickten einander an; aber
dann senkte der Mörder allmählich die Augen zu Boden.

«Ich nehm' euch drei zu Zeugen,» rief der Knabe, die geballte Faust
schüttelnd und im Fortreden einen immer heftigeren Ton annehmend,
«ich nehm' euch drei zu Zeugen, daß ich mich nicht vor ihm fürchte,
und wird er hier gesucht, so zeig' ich ihn selbst an. Ich sag's euch
rund heraus, er kann mich totschlagen, wenn's ihm beliebt, oder wenn
er's wagt, aber bin ich hier, so zeig' ich ihn selbst an, würd' ihn
anzeigen, und wenn er lebendig geröstet werden sollte. Hilfe! Mörder!
Wenn ihr nicht alle drei elende Memmen seid, so steht ihr mir bei.
Hilfe! Mörder! Nieder mit ihm!»

Er warf sich bei diesen Worten allein auf den riesenstarken Mann, und
zwar so wütend und plötzlich, daß beide zu Boden stürzten. Die drei
Zuschauer waren wie betäubt, machten nicht einmal Miene, sich in das
Mittel zu legen, und der Knabe und der Mann wälzten sich um und um,
indem jener der auf ihn herabregnenden Streiche nicht achtete, die
Kleider des Mörders immer fester vor der Brust desselben faßte, und
nicht aufhörte, aus aller Macht nach Hilfe zu rufen.

Der Kampf war jedoch zu ungleich, um lange währen zu können. Sikes
hatte seinen Gegner unter sich gebracht und setzte ihm das Knie auf
die Kehle, als ihn Crackit mit bestürzter Miene emporriß und nach dem
Fenster hinwies. Es schimmerten Lichter auf der Straße unten, eifrige
Stimmen ertönten, von der nächsten Brücke her wurde der unaufhörliche
Schall von Fußtritten vernommen, wie wenn eine zahllose Menschenmenge
herüberkäme, unter welcher sich ein Berittener zu befinden schien,
denn man hörte deutlich das Geräusch von Rossehufen auf dem unebenen
Steinpflaster. Es wurde immer heller, der Nahenden Anzahl immer größer,
und endlich wurde laut an die Haustür geklopft, während ein heiseres
Gemurmel unzähliger zorniger Stimmen auch wohl den Beherztesten mit
Beben erfüllte.

«Hilfe, zu Hilfe!» schrie der Knabe im durchdringendsten Tone. «Hier
ist er, hier ist er! Brecht die Tür auf!»

«In des Königs Namen!» wurde draußen gerufen und wiederum erhob sich,
nur lauter, das zornige Gemurmel.

«Schlagt die Tür ein!» schrie Charley. «Sie öffnen sie nimmermehr.
Schlagt die Tür ein und dann herauf, wo das Licht ist!»

Ein lautes Hussa ertönte, und es war, als wenn mit hundert und abermals
hundert Knütteln und Stangen gegen die Fensterläden gehämmert würde.

«Macht mir das Loch da auf, daß ich diesen verfluchten kleinen
schreienden Galgenstrick einschließen kann», rief Sikes wütend,
schleuderte den Knaben in ein Gemach hinein, das Toby öffnete, und
verschloß es. «Ist die Tür unten gut verwahrt?»

«Verschlossen und doppelt und dreifach verriegelt», erwiderte Crackit,
der gleich den andern beiden kaum einen deutlichen Gedanken fassen zu
können schien.

«Wie ist's mit den Wänden und Fenstern?» fragte Sikes weiter.

«Verwahrt wie ein Gefängnis.»

Jetzt öffnete Sikes das Fenster und rief trotzig hinunter: «Seid alle
verdammt! Macht eure Sachen, so gut ihr könnt, ihr bekommt mich doch
nicht!»

Erschütterndes Geschrei der wütenden Menge erfüllte die Luft. Einige
riefen, man möge das Haus anzünden, andere, die Polizeidiener möchten
den Mörder totschießen, und niemand zeigte eine solche unsinnige
Wut wie der Reiter, der aus dem Sattel sprang, sich durch die Menge
hindurchdrängte, als wenn er nur Wasser teilte, und dicht vor dem
Hause mit einer den gräßlichen Lärm übertönenden Stimme rief: «Zwanzig
Guineen, wer eine Leiter bringt.»

Und nunmehr riefen Hunderte nach Leitern und Schmiedehämmern, andere
rannten mit Fackeln hin und her, und noch andere stießen Flüche und
Verwünschungen aus, drängten wie rasend gegen die Tür oder versuchten
an dem Hause emporzuklimmen.

«Es war Flutzeit, als ich kam», rief Sikes, das Fenster wieder
verschließend. «Gebt mir 'nen Strick. Sie sind alle vorn. Ich kann mich
hinten in den Graben 'nunterlassen und entkommen. 'nen Strick -- hurtig
-- oder ich tue noch drei Mordtaten und mache mir dann selber den
Garaus.»

Die drei von dem Schrecken Gelähmten wiesen nach einem Winkel hin, in
welchem Stricke lagen. Sikes wählte hastig den stärksten und längsten
aus, eilte hinauf und bestieg das Dach.

Der eingesperrte Knabe hatte unterdes nicht aufgehört zu schreien
und zu rufen, man möchte das Haus von allen Seiten bewachen. Als der
Mörder daher aus der Dachluke herausstieg, wurde er sogleich bemerkt,
da Hunderte bereits Wege gesucht hatten, um nach dem Hinterhause zu
gelangen. Die Ebbe war eingetreten, und er sah, daß der Graben nur mit
Schlamm gefüllt war. Die Fenster und Dächer aller Hinterhäuser umher
waren bereits lebendig, und von oben und unten und allen Seiten ertönte
Triumphgeschrei, daß er nicht entrinnen könne.

«Nun haben sie ihn, hurra!» schrie ein Mann auf der nächsten, unter der
Menschenwucht sich beugenden Brücke, und ein tausendfaches Hurra hallte
durch die Luft wider.

«Ich gelobe demjenigen fünfzig Pfund,» rief ein alter, gleichfalls
auf der Brücke stehender Herr, «der ihn lebendig greift. Ich will
hierbleiben und zahle die Summe auf der Stelle.»

Ein abermaliges allgemeines Geschrei ertönte, in das sich der Ruf
mischte, die Tür sei endlich erbrochen; der Menschenstrom flutete
nun wieder zu dieser zurück, denn jeder wollte den Mörder von den
Polizeibeamten herausbringen sehen. Es entstand das furchtbarste
Gedränge, und das Dach wurde für den Augenblick weniger beachtet.

Der Mörder, der, bereits verzweifelnd, unschlüssig dagesessen hatte,
faßte jetzt wieder Hoffnung und beschloß, den letzten Rettungsversuch
zu wagen und sich auf die Gefahr, im Schlamme zu ersticken, in den
Graben hinabzulassen, um womöglich mit Hilfe der Dunkelheit und
Verwirrung zu entfliehen. Die Hoffnung gab ihm neue Kraft, der sich
ihm nähernde Lärm im Hause stachelte ihn noch mehr an, er sprang auf,
erreichte in zwei Augenblicken den Schornstein, befestigte das eine
Ende seines Strickes an demselben und hatte im Nu an dem andern eine
starke Laufschlinge geknüpft. Er konnte sich mit dem Stricke fast bis
auf Manneslänge hinunterlassen und nahm sein Messer zur Hand, um ihn
zur rechten Zeit abzuschneiden und sich in den Graben zu werfen.

In demselben Augenblicke, als er die Schlinge über den Kopf warf, um
sie unter den Armen zu befestigen, und indem der erwähnte alte Herr
laut rief, der Mörder sei im Begriff, sich hinunterzulassen, blickte er
hinter sich, schlug die Hände über dem Kopfe zusammen und stieß einen
lauten Schrei des Entsetzens aus. «Die Augen -- da sind sie wieder!»
rief er mit hohler Grabesstimme, wankte, wie von einem Blitzstrahle
getroffen, verlor das Gleichgewicht und taumelte vom Dache herunter,
die Schlinge war an seinem Halse, und seine Schwere bewirkte, daß sie
straff wie eine Bogenschnur und schnell wie ein Pfeil hinauflief. Er
fiel fünfunddreißig Fuß -- ein plötzlicher Ruck -- ein krampfhaftes
Gliederzucken -- und da hing er mit dem offenen Messer in der
zusammengepreßten, steif werdenden Hand.

Der alte Schornstein bebte von der Erschütterung, hielt sie jedoch
aus. Der entseelte Mörder schwebte hin und wieder; Charley, dem er
die Aussicht versperrte, stieß ihn zur Seite und rief, daß man seiner
Gefangenschaft ein Ende machen möchte; der Hund lief mit schrecklichem
Geheul auf dem Dache hin und her und sprang endlich hinunter auf die
Schulter des Erhängten, vermochte sich aber nicht festzuhalten, stürzte
und lag gleichfalls darauf tot da, denn er war mit dem Kopfe gegen
einen spitzen Stein gefallen.




51. Kapitel.

    Enthüllung mehr als eines Geheimnisses und ein Heiratsantrag ohne
    Erwähnung eines Leibgedinges oder Nadelgeldes.


Zwei Tage nach den im vorigen Kapitel erzählten Ereignissen befanden
sich Mrs. Maylie und Rose, Oliver und der gute Doktor, Mr. Brownlow und
Mrs. Bedwin und noch jemand auf der Reise nach Olivers Geburtsstadt.
Oliver vermochte nur schwer seine Gedanken zu sammeln, und den übrigen
erging es keineswegs besser. Brownlow hatte ihn und die beiden Damen
mit den Monks abgepreßten Aussagen genau bekannt gemacht; und obwohl
sie wußten, daß der Zweck ihrer Reise in der Vollendung des so
glücklich angefangenen Werkes bestand, so war doch die ganze Sache noch
in ein so beträchtliches Dunkel gehüllt, daß die größte Spannung sie
folterte, obgleich Brownlow und Losberne die Schreckensauftritte der
letzten Tage für jetzt noch verborgen vor ihnen gehalten hatten; und so
setzten sie denn ihre Reise schweigend miteinander fort.

Sie gelangten währenddessen auf die Straße, auf welcher Oliver
einst entflohen war, und mit Lebhaftigkeit erneuerte sich ihm die
Erinnerung an jene Leidenszeit. «Sehen Sie, da, da!» rief er in der
höchsten Erregtheit, Roses Hand ergreifend und aus dem Wagenfenster
hinauszeigend. «Das ist der Steg, über den ich hinübersprang, und das
ist die Hecke, hinter welcher ich fortschlich, und das ist der Fußpfad,
der nach dem Hause führt, wo ich mich als kleines Kind aufhielt. O
Dick, mein lieber Dick, wenn ich dich doch jetzt auch sehen könnte!»

«Du wirst ihn bald sehen», sagte Rose; «sollst ihm sagen, wie glücklich
du geworden bist und wie dich nichts so sehr freute, als daß du
gekommen seiest, um ihn an deinem Glücke teilnehmen zu lassen.»

«Ja, ja! Und wir wollen ihn mit uns fortnehmen, ihn kleiden, und er muß
an einen guten Ort, wo er stark und gesund werden kann -- nicht wahr?»

Rose nickte bejahend, denn der Knabe lächelte durch so selige Tränen,
daß sie keines Wortes mächtig war.

«Sie werden freundlich und liebevoll zu ihm sein,» fuhr Oliver fort,
«denn Sie sind es gegen jedermann. Ich weiß es, Sie werden weinen
müssen, wenn Sie hören, was er Ihnen wird erzählen können, aber auch
wieder lächeln, wie Sie es bei mir taten, als ich so ganz anders
geworden war. Er sagte, als ich fortlief: >Geh mit Gott, Gottes Segen
begleite dich!< oh, und ich will nun sagen: >Gottes Segen sei mit dir<,
und ihm zeigen, wie lieb ich ihn dafür habe!»

Als sie endlich durch die engen Straßen der Stadt fuhren, war Oliver
wie außer sich. Da war des Leichenbestatters Haus, nur weit kleiner und
lange nicht so stattlich, wie es ihm vormals erschienen war, und alle
wohlbekannten Gebäude und Läden und Gamfields Karren wie sonst vor dem
Gasthause, und das Armenhaus, das traurige Gefängnis seiner Kinderzeit,
mit den düsteren Fenstern, und am Tore stand derselbe hagere Pförtner.
Oliver schreckte unwillkürlich zurück, lachte über sich selbst, so
töricht zu sein, und weinte und lachte aber- und abermals. So viele
Gesichter an den Türen und Fenstern erkannte er wieder; es war ihm,
als wenn er die Stadt erst gestern verlassen hätte und als wenn seine
letzte Zeit nur ein glücklicher Traum gewesen wäre.

Allein sie und die Gegenwart waren Wirklichkeit. Die Reisenden fuhren
vor dem ersten Gasthause vor, das Oliver einst wie einen Palast
angestaunt hatte, und Mr. Grimwig empfing sie dienstbeflissen, küßte
die alte und junge Dame und war lauter Lächeln und Freundlichkeit, als
wenn er aller Großvater wäre und nicht von fern daran dächte, seinen
Kopf aufzuessen, nicht einmal, als er einem sehr alten Postillon
widersprach und es besser zu wissen behauptete, welcher Weg der nächste
nach London wäre, obwohl er denselben nur ein einziges Mal gekommen,
und obendrein im Schlafe. Die Zimmer und das Mittagessen standen
bereit, und alles war wie durch Zauber geordnet.

Man kleidete sich um, kam wieder zusammen, und dieselbe Stille und
Zurückhaltung begann wieder zu herrschen. Brownlow erschien nicht beim
Mittagessen; die beiden andern Herren liefen mit wichtigen Mienen aus
und ein und flüsterten miteinander, wenn sie im Zimmer waren. Endlich
wurde auch Mrs. Maylie hinausgerufen und kehrte erst nach einer Stunde
mit rotgeweinten Augen zurück. Dieses alles versetzte Rose und Oliver,
die in die neuesten Geheimnisse nicht eingeweiht waren, in große
Unbehaglichkeit und Unruhe. Sie saßen stumm da, und wenn sie bisweilen
ein paar Worte sprachen, so geschah es flüsternd, als ob sie den Laut
ihrer eigenen Stimmen fürchteten.

Es war neun Uhr geworden, und sie fingen an zu glauben, daß sie vor
morgen nichts mehr hören würden, als die Herren Losberne, Grimwig und
Brownlow mit einem Manne hereintraten, bei dessen Erblicken Oliver im
Begriff war laut aufzuschreien. Sie sagten ihm, es wäre sein Bruder,
und es war derselbe, den er in dem Städtchen mit Fagin am Fenster
seines kleinen Zimmers gesehen hatte. Monks oder Charles Leeford setzte
sich unweit der Tür und konnte sich auch jetzt nicht enthalten, dem
erstaunten Knaben einen Blick giftigen Grolls zuzuwerfen. Brownlow
trat mit Papieren in der Hand an den Tisch, in dessen Nähe Rose und
Oliver saßen.

«Diese in London vor mehreren Herren unterzeichneten Erklärungen», hub
er an, «müssen im wesentlichen hier wiederholt werden, so peinlich
es allen Beteiligten auch sein mag. Ich hätte Ihnen die Demütigung
gern erspart, allein es ist notwendig, daß wir Ihre Aussage aus Ihrem
eigenen Munde hören, und Sie wissen, warum.»

«Fahren Sie fort», sagte Monks, sich abwendend. «Rasch. Ich habe genug
getan. Halten Sie mich nicht auf!»

«Dieser Knabe», sprach Brownlow weiter, die Hand auf Olivers Kopf
legend, «ist Ihr Halbbruder, der Sohn Ihres Vaters, meines teuren
Freundes Edwin Leeford, von der jungen Agnes Fleming, der die Geburt
des Kindes das Leben kostete.»

«Ja», sagte Monks, dem zitternden Knaben, dessen Herzschläge er fast zu
hören meinte, fortwährend finster-grollende Blicke zuwerfend. «Er ist
der Bastard.»

«Der Ausdruck, dessen Sie sich bedienen», entgegnete Brownlow im Tone
strengen Tadels, «enthält einen Vorwurf gegen Verstorbene, die den
kurzsichtigen Richtersprüchen dieser Welt längst entrückt sind, und
beschimpft keinen Lebenden, Sie selbst ausgenommen. Doch genug davon.
Der Knabe ist in dieser Stadt geboren?»

«Im Armenhause dieser Stadt. Sie haben es da aufgezeichnet.»

«Die hier Anwesenden müssen es auch hören.»

«So hören Sie. Als sein Vater in Rom erkrankt war, begab sich seine
Frau, meine Mutter, zu ihm -- soviel ich weiß, um sein Vermögen an
sich zu nehmen, denn sie hatte keine Zuneigung zu ihm, wie er nicht
zu ihr. Sie nahm mich mit. Er kannte uns nicht. Denn er lag schon
ohne Bewußtsein und schlummerte bis zum folgenden Tage fort, an dem
er starb. In seinem Schreibtische befand sich ein Päckchen Papiere,
datiert vom ersten Tage seiner Krankheit und adressiert an Sie, Mr.
Brownlow, mit der Bemerkung, daß es erst nach seinem Tode zu befördern
sei. Das Päckchen enthielt ein Schreiben an Agnes Fleming und ein
Testament. Das Schreiben war voll von reuigen Bekenntnissen seiner
gegen sie angewandten Verführungskünste und Bitten zu Gott um Beistand
für sie. Es fehlten zu der Zeit nur noch ein paar Monate bis zu ihrer
Entbindung. Er sagte ihr, was er zu tun beabsichtigte, ihre Unehre
zu verbergen, wenn er am Leben bliebe, und flehte sie an, falls er
sterbe, seinem Andenken nicht zu fluchen, oder zu glauben, daß sein
und ihr Vergehen an ihr und ihrem Kinde heimgesucht werden würde, denn
die ganze Schuld wäre sein. Er erinnerte sie an den Tag, an welchem er
ihr das kleine Schloß geschenkt und den Ring mit ihrem Taufnamen und
einem offenen Raume für den Namen, den er gehofft auf sie übertragen
zu können; bat sie, das Geschmeide, wie sonst, auf ihrem Herzen zu
bewahren, und wiederholte dann das alles aber- und abermals, als wenn
er von Sinnen gewesen wäre -- was, wie ich glaube, auch wirklich der
Fall gewesen ist.»

«Aber das Testament», fiel Brownlow ein, da Oliver schmerzliche Zähren
über die Wangen hinabliefen, «war in demselben Sinne und Geiste
abgefaßt. Er sprach darin von dem Elende, das ihm seine Frau bereitet,
von der frühen Bosheit und Ruchlosigkeit seines einzigen in Haß gegen
ihn erzogenen Sohnes und vermachte Ihnen und Ihrer Mutter Jahrgelder
von je achthundert Pfund. Die Masse seines Vermögens teilte er in zwei
gleiche Teile und bestimmte den einen für -- Agnes Fleming, und den
andern für sein und ihr Kind, wenn es lebendig geboren würde und die
Jahre der Mündigkeit erreichte. Wenn es ein Mädchen wäre, so sollte ihm
die Erbschaft bedingungslos zufallen; wäre es aber ein Knabe, so sollte
sie an die Bedingung geknüpft sein, daß der Erbe bis zu den Jahren
der Mündigkeit seinen Namen durch keinerlei öffentliches Vergehen
befleckte. Ihr Vater traf diese Bestimmung, wie er sagte, um dadurch
sein Vertrauen zu der Mutter und seine, durch den herannahenden Tod nur
verstärkte Überzeugung darzulegen, daß ihr Kind ihre Tugenden, ihre
edlen Gesinnungen erben würde. Wenn seine Voraussetzung nicht einträfe,
sollte das Geld Ihnen zufallen; denn nur, wenn beide Kinder einander
gleich wären, sollte Ihr früherer Anspruch auf sein Vermögen anerkannt
sein, der Sie keine Ansprüche auf sein Herz, und ihm von der frühesten
Kindheit an Kälte und Abneigung bewiesen hätten.»

«Meine Mutter», nahm Monks, und zwar mit lauterer Stimme, wieder das
Wort, «tat, was einer Mutter zukam -- sie verbrannte dieses Testament.
-- Das Schreiben gelangte nie an seine Adresse, sie bewahrte es aber
nebst anderen Dokumenten auf, falls die Flemings den Versuch machen
sollten, den Makel abzuleugnen. Agnes' Vater vernahm die Wahrheit über
sie mit jeder Übertreibung und Vergrößerung, die ihr bitterer Haß --
wofür ich sie jetzt liebe, hinzuzufügen vermochte. Sein verletztes
Ehrgefühl bewog ihn, sich mit seinen Kindern nach einem entlegenen
Orte in Wales zu begeben und, um desto gewisser selbst seinen Freunden
verborgen zu bleiben, sogar seinen Namen zu verändern. Er wurde nicht
lange darauf tot in seinem Bette gefunden. Die Tochter war einige
Wochen zuvor heimlich entwichen. Er hatte selbst die Umgegend nach ihr
durchstreift, war aber mit der Überzeugung zurückgekehrt, daß sie sich
den Tod gegeben, und überlebte seinen Kummer nur wenige Stunden.»

Es trat ein kurzes Stillschweigen ein, bis Brownlow den Faden der
Erzählung wieder aufnahm. «Nach Jahren,» sagte er, «erschien dieses
jungen Mannes -- Eduard Leefords -- Mutter bei mir. Er hatte sie in
seinem achtzehnten Jahre verlassen, sie ihrer Juwelen und ihres Geldes
beraubt, hatte gespielt, vergeudet, gefälscht und war nach London
gegangen, wo er sich dem schlechtesten Gesindel zugesellte. Sie litt
an einer schmerzhaften und unheilbaren Krankheit und wünschte ihn vor
ihrem Tode noch wiederzusehen. Es wurden die genauesten Nachforschungen
angestellt, welche endlich Erfolg hatten. Er ging mit ihr nach
Frankreich zurück.»

«Und sie starb dort,» fiel Monks ein, «nachdem sie lange auf dem
Siechbette gelegen; kurz vor ihrem Tode vermachte sie mir diese
Geheimnisse, samt ihrem unauslöschlichen und tödlichen Hasse gegen
alle in diese Angelegenheit Verwickelten, was jedoch unnötig war, denn
er lebte schon seit langer Zeit in mir. Sie wollte es nicht glauben,
daß das Mädchen sich und dem Kinde den Tod gegeben, sondern hielt
sich überzeugt, daß ein Knabe geboren und am Leben wäre. Ich schwur
ihr, wenn je das Dasein eines solchen zu meiner Kunde gelangte, ihm
nachzuspüren, ihm nimmer Ruhe zu lassen, ihn mit der bittersten,
unversöhnlichsten Feindschaft zu verfolgen, allen Haß an ihm
auszulassen, dessen mein Innerstes fähig war -- ihn, den hochtrabenden
Worten des beleidigenden Testaments zum Hohne, an den Galgen zu
bringen. Sie hatte recht gehabt. Er kam mir endlich in den Weg; ich
machte einen guten Anfang -- und würde -- ja, würde geendet haben,
wie begonnen, wenn nicht eine schwatzmäulige Trulle meine Anschläge
vereitelt hätte!»

Der Schändliche schlug sich mit der Hand vor die Stirn, murmelte in
der Wut ohnmächtiger Bosheit Verwünschungen über sich selbst; Brownlow
wandte sich unterdessen zu seinen entsetzten Freunden und sagte ihnen,
daß der Jude, der Leefords alter Vertrauter und Helfershelfer gewesen
wäre, eine große Belohnung von ihm für Olivers Umstrickung erhalten
hätte. Es sei ausbedungen gewesen, daß er einen Teil der gezahlten
Summe zurückerstatten solle, falls Oliver wieder frei würde, und ein
Streit über diesen Punkt habe beide auf das Land geführt, welche Reise
den Zweck gehabt, zu erkunden, ob der von Mrs. Maylie aufgenommene
Knabe wirklich Oliver sei.

«Was haben Sie von dem Schlosse und Ringe zu sagen?» fragte Brownlow,
Monks wieder anredend.

«Ich kaufte sie den Leuten ab, von welchen ich Ihnen gesagt habe. Sie
hatten sie der Wärterin entwandt, die sie der Leiche abgenommen hatte»,
erwiderte Monks, ohne die Augen aufzuschlagen. «Sie wissen, was daraus
geworden ist.»

Brownlow gab Grimwig einen Wink, dieser eilte voller Eifer hinaus und
kehrte nach wenigen Augenblicken mit dem widerstrebenden Ehepaare aus
dem Armenhause zurück.

«Trügen mich meine Augen, oder sehe ich den kleinen Oliver wirklich
vor mir?» rief Mr. Bumble mit schlecht erkünsteltem Entzücken. «Ach,
Oliver, wenn du wüßtest, wie bekümmert ich um dich gewesen bin!»

«Schweig, Dummkopf!» murmelte seine Ehehälfte.

«Frau, kann ich meinen Gefühlen wehren,» entgegnete er, «wenn ich, der
ich ihn kirchspielmäßig erzogen habe, ihn sitzen sehe zwischen den
allerangenehmsten Damen und Herren? Ich hatte den Knaben immer so lieb,
als wenn er mein eigner -- eigner Großvater gewesen wäre», sprudelte
Mr. Bumble heraus, nachdem er mühsam dem passendsten Vergleich
nachgesonnen hatte. «Master Oliver, mein guter Oliver, erinnerst du
dich noch des lieben, braven Herrn mit der weißen Weste? Ach, er schied
vorige Woche von der Erde in den Himmel, mit einem eichenen Sarge mit
plattierten Griffen, Oliver.»

«Seien Sie so gut, Ihre Gefühle für sich zu behalten, Sir», fiel
Grimwig bissig ein.

«Ich will mein möglichstes tun, Sir», erwiderte Bumble und wandte sich
zu Brownlow: «Wie befinden Sie sich, Sir? Hoffentlich sehr wohl.»

Brownlow beachtete seine Frage nicht, trat dicht vor das Ehepaar, wies
nach Monks und fragte seinerseits: «Kennen Sie den Mann?»

«Nein», antwortete Frau Bumble keck.

«Vielleicht kennen Sie ihn, Mr. Bumble?»

«Ich habe ihn nie in meinem Leben gesehen.»

«Ihm auch nichts verkauft?»

«Nein», sagte Frau Bumble.

«Hatten Sie nicht einmal ein goldenes Schloß und einen Ring?»

«Behüte. Sind wir denn bloß hier, um so läppische Fragen zu
beantworten?»

Brownlow gab Grimwig abermals einen Wink, und abermals enteilte Grimwig
mit ungemeinem Eifer und kehrte mit zwei alten, wankenden, gichtischen
Frauen zurück.

«Sie verschlossen die Tür an dem Abend, da die alte Sally starb,
konnten aber die Ritzen nicht verstopfen», sagte die eine, ihre welke
Hand emporhebend, und die andere stimmte bei.

«Wir hörten,» fuhr die erste fort, «daß sie Ihnen sagen wollte, was
sie getan hatte, und sahen, daß Sie ihr etwas in Papier aus der Hand
nahmen, und am anderen Tage, daß Sie zum Pfandleiher gingen.»

«Ja,» fügte die zweite hinzu, «und wir spürten auch aus, daß in dem
Papiere ein goldenes Schloß und ein Ring gewesen war.»

«Und wissen noch mehr», sprach die erste weiter. «Die alte Sally hat
uns oft erzählt, die junge Frauensperson hätte ihr gesagt, daß sie
gefühlt hätte, sie würde es nicht überleben, und wäre zur Zeit, da sie
den Knaben geboren, auf dem Wege gewesen, um am Grabe des Vaters ihres
Kindes zu sterben.»

«Wollen Sie den Pfandleiher sehen?» fragte Grimwig, mit der Hand auf
dem Türgriffe.

«Nein», antwortete Frau Bumble. «Da er» -- sie wies nach Monks -- «so
memmenhaft gewesen ist, zu bekennen, wie ich sehe, daß er es gewesen,
und da Sie aus all den alten Hexen die rechten herausgespürt, so habe
ich nichts mehr zu sagen. Ja, ich verkaufte die alten Scharteken; sie
liegen, wo Sie sie nimmer wiederfinden werden, und was nun mehr?»

«Oh, nichts weiter,» sagte Brownlow, «als daß es jetzt unsere Sache
ist, Sorge zu tragen, daß man Ihnen und Ihrem Manne kein Vertrauen als
Beamten mehr schenkt. Sie können gehen.»

«Ich will doch hoffen,» nahm Bumble bestürzt das Wort, als Grimwig die
beiden alten Frauen hinausführte, «daß mich dieser unglückliche kleine
Umstand meines Kirchspieldienstes nicht berauben wird?»

«Das wird er allerdings,» erwiderte Brownlow, «und Sie können sehr froh
sein, wenn Sie noch so davonkommen.»

Frau Bumble entfernte sich, und sobald sie die Tür hinter sich
geschlossen hatte, erklärte ihr Eheherr, daß sie alles getan und sich
davon nicht hätte zurückhalten lassen wollen.

«Das ist keine Entschuldigung», sagte Brownlow. «Sie waren gegenwärtig
bei dem Verkaufe und sind vor dem Gesetze der noch schuldigere Teil, da
Ihre Frau gemäß demselben unter Ihrer Leitung handelt.»

«Wenn das Gesetz so lautet,» sagte Bumble, seinen Hut pathetisch
zusammendrückend, «so ist es ein Esel -- ein Einfaltspinsel. Wenn es so
kurzsichtig ist, so ist's ein bloßer Junggesell, und ich wünsche ihm
das Allerärgste -- nämlich, daß ihm die Augen durch Erfahrung geöffnet
werden mögen -- ja, durch Erfahrung!»

Er folgte nach diesen Worten seiner Ehehälfte mit verzweifelter
Resignation, und Brownlow wandte sich zu Rose.

«Mein liebes Fräulein, reichen Sie mir Ihre Hand. Zittern Sie nicht;
Sie können die wenigen Worte, welche wir noch zu sagen haben, ohne
Furcht hören.»

«Ich weiß nicht, ob sie Bezug auf mich haben können,» sagte Rose, «aber
wenn -- wenn es der Fall ist, so lassen Sie sie mich ein andermal
hören. Ich habe jetzt nicht die Kraft dazu.»

«Sie sind stärker, als Sie glauben», wandte Brownlow ein; «ich weiß es.
Kennen Sie diese junge Dame, Sir?»

Monks bejahte.

«Ich sah Sie nie», sagte Rose mit bebender Stimme.

«Ich habe Sie oft gesehen», versetzte Monks.

«Der unglücklichen Agnes' Vater hatte zwei Töchter», fiel Brownlow ein.
«Was war das Schicksal der anderen -- der jüngsten?»

«Als ihr Vater starb,» antwortete Monks, «an einem fremden Orte,
unter angenommenem Namen, ohne das mindeste zu hinterlassen, was zur
Auffindung ihrer Verwandten hätte führen können, nahmen arme Leute
sie zu sich und erzogen sie. Der Haß spürt nicht selten auf, was der
Liebe verborgen bleibt. Meine Mutter fand das Kind nach Jahresfrist.
Die Leute waren arm und fingen an, ihrer aufopfernden Großmut müde zu
werden. Zum wenigsten war es bei dem Manne der Fall. Meine Mutter ließ
ihnen das Mädchen daher, gab ihnen ein unbedeutendes Geschenk an Geld
und versprach mehr, was sie aber nie zu schicken gedachte. Die Armut
und Unzufriedenheit der Leute verhießen, daß das Kind unglücklich genug
werden würde, schienen meiner Mutter indes noch nicht ganz zu genügen.
Sie erzählte ihnen daher die Geschichte der Schwester mit angemessenen
Veränderungen und sagte ihnen, sie möchten auf das Kind sorgfältig
achten, denn es stamme von bösem Blute her, wäre unehelich geboren und
würde früher oder später auf üble Wege geraten. Die Umstände schienen
das alles zu bestätigen, die Leute glaubten es und behandelten das Kind
so hart, wie wir es nur wünschen konnten, bis der Zufall wollte, daß
eine damals in Chester wohnende verwitwete Dame aus Mitleid es mit sich
fortnahm. Es war, als wenn ein Höllenspuk uns genarrt hätte, denn trotz
all unserer Anstrengungen blieb des alten Fleming Tochter bei der Dame
und war glücklich; ich verlor sie vor ein paar Jahren aus den Augen und
sah sie erst vor wenigen Monaten wieder.»

«Sehen Sie die junge Dame jetzt?»

«Ja -- sie lehnt an Ihrem Arme.»

Rose war einer Ohnmacht nahe. Mrs. Maylie schloß sie in die Arme und
rief aus: «Du bist und bleibst meine liebe Nichte -- mein über alles
teures Kind. Ich möchte dich um alle Schätze der Welt nicht verlieren.»

«Sie sind die einzige Freundin, die ich jemals hatte,» schluchzte Rose,
«sind mir stets die liebreichste, beste Mutter gewesen. O wie soll ich
dieses alles ertragen!»

«Du hast mehr erduldet und hast dich unter jeglichem Leid als das
beste, herrlichste Mädchen gezeigt und von jeher alle froh und
glücklich gemacht, die dich kannten. Aber schau hier, wer es ist, der
sich sehnt, dich in die Arme zu schließen.»

«Oh, ich werde sie niemals Tante nennen», rief Oliver. «Meine
Schwester, meine liebe Schwester. Es war etwas in meinem Herzen, das
mich von Anfang an trieb, sie so innig zu lieben. O Rose, meine beste
Rose!»

Mögen die Tränen, welche geweint, die abgebrochenen Worte, die in der
Umarmung der beiden Waisen gewechselt wurden, geheiligt sein! Ein
Vater, eine Schwester und Mutter waren in demselben Augenblick gewonnen
und verloren; Freude und Schmerz gemischt in der Schale; doch war keine
Zähre eine bittere, denn auch der Schmerz war so gemildert, und so
süße, wonnige Gedanken gesellten sich ihm, daß er in eine hohe Freude
verwandelt wurde und ganz seinen Stachel verlor.

Sie waren eine lange, lange Zeit allein. Endlich wurde leise geklopft,
Oliver öffnete die Tür, schlich hinaus und Harry Maylie stand im Zimmer.

«Ich weiß alles», sagte er, neben der lieblichen Jungfrau Platz
nehmend. «Teure Rose, ich weiß alles, -- wußte es gestern schon --
und komme, dich an ein Versprechen zu erinnern. Du gabst mir die
Erlaubnis, jederzeit innerhalb eines Jahres auf den Gegenstand unserer
letzten Unterredung zurückzukommen -- nicht in dich zu dringen, deinen
Entschluß zu ändern, dich ihn wiederholen zu hören, wenn du wolltest.
Ich sollte dir zu Füßen legen dürfen, was ich besäße, nur ohne den
Versuch zu machen, wenn du bei deinem Beschlusse beharrtest, ihm untreu
zu werden.»

«Dieselben Gründe, welche mich damals bestimmten, bestimmen mich noch
jetzt», erwiderte Rose mit Festigkeit. «In welchem Augenblicke könnte
ich lebhafter empfinden, was ich der edlen Frau schuldig bin, die mich
von einem leiden- und vielleicht schmachvollen Leben errettet hat? Ich
habe einen Kampf zu kämpfen, bin aber stolz darauf, ihn zu bestehen; er
ist ein schmerzlicher, aber mein Herz wird nicht erliegen.»

«Die Enthüllungen dieses Abends --»

«Lassen mich in bezug auf dich in derselben Lage.»

«Du verhärtest dein Herz gegen mich, Rose.»

«O Harry, Harry,» sagte Rose, in Tränen ausbrechend, «ich wollte, daß
ich es könnte, um mir diese Pein zu ersparen.»

«Warum aber fügst du sie dir selber zu?» entgegnete Harry, ihre Hand
ergreifend. «Denk doch an das, was du heute abend vernommen, Rose!»

«Ach, was habe ich vernommen! Daß mein Vater den ihm zugefügten Schimpf
tief genug empfand, um sich in gänzliche Verborgenheit zurückzuziehen
-- o Harry, wir haben genug gesagt.»

«Noch nicht, noch nicht», rief er, die Aufstehende zurückhaltend.
«Meine Hoffnungen, Wünsche, Entwürfe, Gefühle -- alles in mir ist
anders geworden, nur meine Liebe nicht. Ich biete dir jetzt keine
Auszeichnung, keine glänzende Stellung mehr in einer verkehrten,
trugvollen Welt, in welcher alles beschimpft, nur das wahrhaft
Schandbare nicht; nein, nur einen stillen, bescheidenen häuslichen
Herd, liebste Rose, mehr habe ich nicht zu bieten.»

«Was willst du damit sagen?» stammelte die junge Dame.

«Als ich das letztemal von dir schied, verließ ich dich mit dem
festen Entschlusse, alle eingebildeten Schranken zwischen dir und mir
niederzureißen -- deine Welt zur meinigen zu machen, wenn die meinige
nicht die deine sein könnte -- und dem Geburtsstolze den Rücken zu
wenden, damit er nicht hochmütig auf dich herabzuschauen vermöchte. Ich
habe es getan. Die mich an sich zogen, entfernten sich von mir -- die
mich anlächelten, zeigen mir frostige Mienen. Wohl! es gibt lachende
Gefilde und schattige Bäume in Englands schönster Grafschaft, dort
neben einer Dorfkirche -- der meinigen, Rose, steht ein ländliches
Haus, auf das du mich stolzer machen kannst, als es alle die Hoffnungen
und Aussichten vermögen, denen ich entsagt habe, entsagt haben würde,
und wenn sie noch tausendmal lockender gewesen wären. Das ist jetzt
mein Besitztum und mein Stand, meine Stellung in der Welt -- und ich
lege alles vor dir nieder.»

       *       *       *       *       *

«'s ist 'ne Geduldsprobe, mit dem Abendessen auf Verliebte zu warten»,
sagte Grimwig, aus einem Schläfchen erwachend.

Die Wahrheit zu sagen, das Abendessen ließ ungebührlich lange auf
sich warten, und weder Mrs. Maylie noch Harry oder Rose (die zugleich
erschienen) wußten auch nur ein Wort zur Entschuldigung zu sagen.

«Ich dachte ernstlich daran, heute abend meinen Kopf aufzuessen,» sagte
Grimwig, «denn ich fing an zu glauben, daß ich weiter nichts bekommen
würde. Wenn Sie erlauben, so nehme ich mir die Freiheit, die angehende
Braut zu küssen.»

Er verlor keine Zeit, seine Ankündigung bei dem errötenden Mädchen zur
Ausführung zu bringen, und sein Beispiel ermunterte den Doktor und
Brownlow zur Nachfolge. Einige wollen wissen, Harry Maylie hätte es
selbst in einem anstoßenden dunkeln Zimmer gegeben, was jedoch von den
besten Autoritäten für arge Verleumdung erklärt wird, da er jung und
ein Geistlicher gewesen wäre.

«Mein lieber Oliver, wo bist du gewesen, und warum siehst du so traurig
aus?» fragte Mrs. Maylie. «Wie -- Tränen in diesem Augenblicke?»

Wir leben in einer Welt der Täuschungen. Wie oft sehen wir unsere
liebsten -- die am meisten uns ehrenden Hoffnungen vereitelt!

Der arme kleine Dick war tot.




52. Kapitel.

    Des Juden letzte Nacht.


Der Gerichtssaal war zum Ersticken gefüllt -- kein Auge, das nicht
auf den Juden geheftet gewesen wäre. Der Vorsitzende erteilte den
Geschworenen die Rechtsbelehrung. Mit größter Spannung horchend stand
Fagin da, die Hand am Ohre, um kein Wort zu verlieren. Bisweilen
blickte er scharf nach den Geschworenen hinüber, die Wirkung auch
nur des im kleinsten ihm günstigen Worts zu erlauschen -- bisweilen
angstvoll nach seinem Anwalt, wenn die Rede in erschütternder,
furchtbarer Klarheit wider ihn zeugte. Sonst aber regte er weder Hand
noch Fuß und verharrte noch in der Stellung des angstvoll Horchenden,
nachdem der Richter seine Darlegung längst beendigt.

Ein leises Gemurmel rief ihn zum Bewußtsein zurück. Er hob die Augen
empor und sah die Geschworenen miteinander beraten. Alle Blicke waren
auf ihn gerichtet, und man flüsterte schaudernd miteinander. Einige
wenige schienen ihn nicht zu beachten. In ihren Mienen drückte sich
unruhige Verwunderung aus, wie die Jury zögern könne, ihr Schuldig
auszusprechen; allein in keinem Antlitze -- sogar in keinem der
zahlreich anwesenden Frauen -- vermochte er auch nur das leiseste
Anzeichen des Mitleids zu lesen. Alle schienen mit Begier seine
Verurteilung zu fordern.

Abermals trat eine Totenstille ein -- die Geschworenen hatten sich an
den Vorsitzenden gewandt. Horch!

Sie baten nur um die Erlaubnis, sich zurückziehen zu dürfen.

Er forschte, als sie einer hinter dem andern hinausgingen, in ihren
Mienen, wohin wohl die Mehrzahl neigen möchte; allein vergeblich. Der
Kerkermeister berührte ihn an der Schulter. Er folgte ihm mechanisch in
den Hintergrund der Anklagebank und ließ sich auf einen Stuhl nieder,
den jener ihm wies, denn er würde ihn sonst nicht gesehen haben.

Er schaute abermals nach den Zuhörern. Einige aßen und andere wehten
sich mit den Tüchern Kühlung zu. Ein junger Mann zeichnete sein Gesicht
in eine Brieftasche. Fagin dachte, ob die Zeichnung wohl ähnlich werden
möchte, und sah zu, als der Zeichner seinen Bleistift spitzte, wie es
jeder unbeteiligte Zuschauer hätte tun können.

Er wandte sich nach dem Richter und begann sich innerlich mit dem
Anzuge desselben zu beschäftigen -- von welchem Schnitte er wäre und
was er kosten dürfe. Auf der Richterbank hatte ein alter, beleibter
Herr gesessen, der sich entfernt hatte und jetzt zurückkehrte, und
Fagin überlegte, ob der Herr zu Mittag gespeist und wo er gesessen,
und was dergleichen Gedanken mehr waren, bis ein neuer Gegenstand neue
Gedanken in ihm erweckte.

Trotzdem war freilich sein Gemüt keinen Augenblick von dem peinigenden
und drückenden Gefühle frei, daß sich das Grab zu seinen Füßen öffnete;
es schwebte ihm fortwährend vor, aber undeutlich und unbestimmt, und er
vermochte seine Gedanken nicht dabei festzuhalten. Und so geschah es,
daß er, während bald Fieberhitze ihn ergriff und es ihn bald mit kaltem
Schauder überlief, die eisernen Stäbe der Anklagebank zählte, die er
vor sich sah, und bei sich selber dachte, wie es wohl gekommen sein
möchte, daß einer derselben abgebrochen wäre; und ob man wohl einen
neuen einschlagen würde oder nicht. Dann schweiften seine Gedanken
wieder zu den Schrecken des Galgens und Schafotts ab, bis ein Aufwärter
den Boden mit Wasser besprengte, was jenen abermals eine andere
Richtung gab.

Endlich wurde Stille geboten, und alle Blicke waren plötzlich auf die
Tür gerichtet. Die Geschworenen kehrten zurück und gingen dicht an
ihm vorüber. Die Gesichter der Geschworenen waren wie von Stein, er
vermochte nichts darin zu lesen. Es trat eine Stille ein -- vollkommen
-- atemlos -- Schuldig!

Der Saal hallte von einem erschütternden Rufe wider, der sich mehrmals
wiederholte und durch ein donnerndes Geschrei beantwortet wurde, durch
welches die Menge draußen ihren Jubel ausdrückte, daß der Verurteilte
am Montag sterben müsse.

Er wurde gefragt, ob er etwas zu sagen wisse, weshalb die
Urteilsvollziehung nicht statthaben möchte. Er hatte seine horchende
Stellung wieder angenommen und blickte den Richter scharf an, der
jedoch die Frage zweimal wiederholen mußte, ehe der Jude sie zu
vernehmen schien, der endlich nur murmelte, er wäre ein alter Mann --
ein alter Mann -- ein alter Mann. Seine Stimme verlor sich in leises
Flüstern, und bald schwieg er gänzlich.

Der Richter setzte die schwarze Mütze auf, -- der Verurteilte stand
noch immer da mit derselben Miene in derselben Stellung. Die ernste
Feierlichkeit des Augenblicks preßte einer Frau einen Ausruf aus --
er blickte hastig und lauschend empor -- stand aber da gleich einer
Bildsäule, obgleich der Ton, das Wort, alle Anwesenden durchbebte. Er
blickte noch immer starr vor sich hin, als ihm der Kerkermeister die
Hand auf den Arm legte und ihm winkte. Er sah ihn einen Augenblick wie
betäubt an und gehorchte.

Er wurde hinunter in einen gepflasterten Raum geführt, wo einige
Angeklagte warteten, bis die Reihe an sie käme, und andere sich mit
ihren Freunden unterredeten, die sich an dem in den Hof öffnenden
vergitterten Fenster gesammelt hatten und unter denen niemand war,
der mit ihm gesprochen hätte, die aber alle bei seiner Annäherung
zurücktraten, um ihn der Volksmenge draußen hinter den Eisenstäben
sichtbarer zu machen; und er wurde laut mit Schimpfnamen, Geschrei und
Gezisch begrüßt. Er schüttelte die Faust und würde die Nächststehenden
angespien haben, allein seine Führer drängten ihn rasch fort durch
einen düsteren, nur von wenigen matt brennenden Lampen erleuchteten
Gang in das Innere des Gefängnisses, wo er durchsucht wurde, ob
er nicht etwa an seiner Person die Mittel hätte, dem Gesetze
vorzugreifen. Endlich brachten sie ihn in eine der Armesünderzellen und
ließen ihn darin -- allein.

Er setzte sich der Tür gegenüber auf eine steinerne Bank, die als Sitz
und Lager diente, heftete die blutunterlaufenen Augen auf den Boden
und bemühte sich, seine Gedanken zu sammeln. Nach einiger Zeit begann
er sich einzelner Bruchstücke der Anrede des Richters zu erinnern,
obwohl es ihm, während sie gesprochen worden, gewesen war, als wenn
er kein Wort hören könnte. Ein Teil fügte sich allmählich zum andern,
und endlich stand das Ganze fast vollständig klar vor ihm. Aufgehängt
zu werden am Halse, bis er tot wäre -- das war das Ende gewesen.
Aufgehängt zu werden, bis er tot wäre.

Es wurde dunkel, sehr dunkel, und er fing an, aller derer zu gedenken,
die er gekannt und die auf dem Schafott gestorben waren -- einige durch
seine Schuld oder auf seinen Betrieb. Sie tauchten in so rascher Folge
vor ihm auf, daß er sie kaum zu zählen vermochte. Er hatte mehrere von
ihnen sterben sehen -- und sie verspottet, weil sie mit Gebeten auf
den Lippen verschieden waren. Wie gedankenschnell sie aus starken,
kräftigen Männern in baumelnde Fleischklumpen verwandelt waren!

Mancher von ihnen hatte vielleicht dasselbe Gemach bewohnt -- auf
derselben Stelle gesessen. Es war sehr finster -- warum wurde kein
Licht gebracht? Die Zelle war vor vielen Jahren erbaut -- Hunderte
mußten ihre letzten Stunden darin verlebt haben -- man saß darin wie
in einem mit Leichen angefüllten Gewölbe -- und viele derselben hatten
wohlbekannte Gesichter -- Licht, Licht!

Endlich, als er sich die Hände an der eisenverwahrten Tür fast blutig
geschlagen hatte, erschienen zwei Männer, deren einer ein Licht trug,
das er auf einen eisernen, in der Mauer befestigten Leuchter steckte,
während der andere eine Matratze hinter sich herzog, um darauf die
Nacht zuzubringen, denn der Gefangene sollte fortan nicht mehr allein
gelassen werden.

Dann kam die Nacht -- die finstere, schauerliche, schweigende Nacht.
Andere Wachende freuen sich, die Kirchglocken schlagen zu hören, die
vom Leben zeugen und den nahenden Tag verkünden. Dem Juden brachten sie
Verzweiflung. Jedes Anschlagen des eisernen Klöppels führte ihn zu dem
einen hohlen Schall -- Tod. Was nützte das Geräusch des geschäftigen,
heiteren Morgens, das selbst in den Kerker und zu ihm drang? Es war
Totengeläute anderer Art, das noch den Hohn zur schrecklichernsten
Mahnung hinzufügte.

Der Tag verging -- Tag! Da war kein Tag; er war so bald entschwunden
wie angebrochen, und abermals kam die Nacht -- Nacht! So lang und
doch so kurz; lang in ihrem schrecklichen Schweigen, und kurz nach
ihren flüchtigen Stunden. Jetzt redete der Elende irre und stieß
Gotteslästerungen aus -- dann heulte er und zerraufte sein Haar.
Ehrwürdige Männer seines Glaubens waren gekommen, mit ihm zu beten,
allein er hatte sie mit Verwünschungen hinausgetrieben. Sie erneuerten
ihre menschenfreundlichen Versuche und mußten seinen gewalttätigen
Drohungen weichen.

Sonnabend -- nur noch eine einzige Nacht! Und während er noch sann und
sann: nur noch eine einzige Nacht! dämmerte es schon -- Sonntag!

Erst am Abend dieses schauervoll-bangen Tages ward seine verpestete
Seele von einem vernichtenden Gefühle ihrer verzweifelten Lage
ergriffen. Nicht, daß er auch nur von fern eine bestimmte Hoffnung,
Gnade zu erlangen, gehegt hätte; er hatte es nur noch nicht über sich
vermocht, den Gedanken, so bald sterben zu müssen, klar und deutlich
auszudenken. Er hatte nur wenig zu den beiden Männern gesprochen,
die sich einander bei ihm ablösten, und sie hatten sich ihrerseits
nicht um ihn gekümmert. Er hatte wachend dagesessen, aber geträumt.
Jetzt sprang er von Minute zu Minute auf und rannte mit keuchendem
Munde und brennender Stirn in entsetzlicher Furcht- und Zorn- und
Grimmanwandlung auf und nieder, daß sie sogar -- die an dergleichen
Gewöhnten -- schaudernd vor ihm zurückbebten. Er wurde zuletzt unter
den Folterqualen seines bösen Gewissens so fürchterlich, daß keiner es
ertragen konnte, allein bei ihm zu sitzen und ihn vor Augen zu haben,
-- daß seine Wärter beschlossen, miteinander Wache bei ihm zu halten.

Er kauerte auf seinem Steinbette nieder und dachte der Vergangenheit.
Er war bei seiner Abführung in das Gefängnis verwundet worden und
trug deshalb ein leinenes Tuch um den Kopf. Sein rotes Haar hing auf
sein blutloses Gesicht herunter; sein Bart war zerzaust und in Knoten
gedreht; aus seinen Augen leuchtete ein schreckliches Feuer; seine
ungewaschenen Glieder bebten von dem in ihm brennenden Fieber. Acht
-- neun, zehn! Wenn man die Glocken vielleicht nicht schlagen ließ,
bloß um ihn mit Schrecken zu erfüllen, wenn sie die einander auf den
Fersen folgenden Stunden wirklich anzeigten -- wo mußte er sein, wenn
sie abermals schlugen? Elf! Noch ein Schlag, ehe die Stimme der letzten
Stunde verklungen war. Um acht Uhr war er, wie er sich sagte, der
einzige Leidtragende zu seinem eigenen Grabgefolge; um elf --

Newgates schreckliche Mauern, die so viel Elend und so unaussprechliche
Angst und Pein nicht bloß vor den Augen, sondern nur zu oft und zu
lange auch vor den Gedanken der Menschen verbargen, umschlossen nie ein
so entsetzliches Schauspiel wie dieses. Die wenigen Vorübergehenden,
die etwa stillstanden und bei sich dachten, was der Verurteilte wohl
vornehmen möchte, der am folgenden Tage hingerichtet werden sollte,
würden die Nacht darauf gar schlecht geschlafen haben, wenn sie ihn im
selben Augenblicke hätten sehen können.

Vom Abend bis fast um Mitternacht traten bald einzelne, bald mehrere
zu dem Pförtner, fragten in großer Spannung, ob ein Aufschub der
Hinrichtung verfügt sei, und teilten die willkommene Verneinung andern,
in Haufen Stehenden mit, die auf die Tür hinwiesen, aus welcher er
kommen müßte, die Stelle zeigten, wo das Schafott errichtet werden
würde, sich widerstrebend entfernten und im Fortgehen die zu erwartende
Szene sich im voraus ausmalten. Endlich waren alle heimgekehrt und die
Straßen umher auf eine Stunde in der Mitte der Nacht der Einsamkeit und
Finsternis überlassen.

Der Raum vor dem Gefängnisse war gesäubert, und man hatte einige
starke, schwarz bemalte Schranken, dem vorauszusehenden großen Gedränge
zu wehren, errichtet, als Mr. Brownlow mit Oliver an dem Pförtchen
erschien und eine Sheriffserlaubnis vorwies, den Verurteilten sehen zu
dürfen. Sie wurden sogleich eingelassen.

«Soll der kleine Herr auch mit hinein, Sir?» fragte der Schließer, der
ihnen zum Führer gegeben war. «'s ist kein Anblick für Kinder, Sir.»

«Freilich nicht, mein Freund», erwiderte Brownlow; «allein was ich bei
dem Manne zu tun habe, hat auch auf den Knaben sehr genauen Bezug, und
da er ihn als glücklichen Frevler gekannt hat, so halte ich es für gut,
daß er ihn auch jetzt sehe, ob es auch einen etwas peinlichen Eindruck
bei ihm hervorbringen mag.»

Die Worte waren leise gesprochen. Der Schließer berührte den Hut,
blickte mit einiger Neugier nach Oliver und ging ihnen voran, zeigte
ihnen das Tor, aus welchem der Verurteilte kommen würde, machte sie
aufmerksam auf das an ihr Ohr dringende Hämmern der das Schafott
erbauenden Zimmerleute und öffnete ihnen endlich die Tür der Zelle des
Juden.

Dieser saß auf seinem Bette, wiegte sich hin und her, und sein Gesicht
glich mehr dem eines eingefangenen Tieres als einem menschlichen
Antlitze. Er gedachte offenbar seines alten Lebens, denn er murmelte,
Brownlow und Oliver sehend und doch nicht sehend, vor sich hin: «Guter
Junge, Charley -- gemacht brav -- und auch Oliver -- ha, ha, ha, Oliver
-- und sieht aus wie ein Junker -- ganz wie ein -- bringt ihn fort --
zu Bett mit dem Buben!»

Der Schließer faßte Oliver bei der Hand und flüsterte ihm zu, daß er
ohne Furcht sein möchte.

«Zu Bett mit ihm!» rief der Jude. «Hört Ihr nicht? Er -- er ist -- ist
an diesem allen schuld. 's ist des Geldes wert, ihn zu erziehen dazu --
Bolters Kehle, Bill; kümmert Euch um die Dirne nicht -- Bolters Kehle,
so tief Ihr könnt schneiden. Sägt ihm ab den Kopf.»

«Fagin», sagte der Schließer.

«Ja, ja», rief der Jude und nahm rasch die lauschende Stellung an, die
er bei seinem Prozesse behauptet hatte. «Ein alter Mann, Mylord; ein
sehr, sehr alter Mann.»

«Hier ist jemand, Fagin, der Euch etwas zu sagen hat -- seid Ihr ein
Mann?» rief ihm der Schließer, ihn schüttelnd und dann festhaltend, in
das Ohr.

«Werd's nicht mehr sein lange», rief der Jude zurück, mit einem
Angesicht aufblickend, das keinen menschlichen Ausdruck mehr hatte --
nur Wut und Entsetzen malte sich darin. «Schlagt sie alle tot! Was
haben sie für ein Recht, mich abzuschlachten?»

Er erkannte jetzt Oliver und Brownlow, wich in die fernste Ecke
seines Sitzes zurück und fragte, was sie an diesen Ort geführt hätte.
Der Schließer hielt ihn fortwährend fest und forderte Brownlow auf,
rasch zu sagen, was er ihm zu sagen hätte, denn er würde mit jedem
Augenblicke schlimmer.

«Es sind Euch gewisse Papiere zu sicherer Aufbewahrung anvertraut
worden, und zwar von einem Menschen, namens Monks», sagte Brownlow,
sich ihm nähernd.

«'s ist gelogen -- ich habe keine, keine, keine!» erwiderte der Jude.

«Um der Liebe Gottes willen,» sagte Brownlow feierlich, «sprecht nicht
so am Rande des Grabes, sondern sagt mir, wo ich sie finden kann.
Ihr wißt, daß Sikes tot ist, daß Monks gestanden hat, daß Ihr keine
Hoffnung eines Gewinnes mehr habt. Wo sind die Papiere?»

«Oliver,» rief der Jude, dem Knaben winkend, «komm, laß mich dir
flüstern ins Ohr.»

«Ich habe keine Furcht», sagte Oliver leise zu Brownlow und ging zu
dem Juden, der ihn zu sich zog und ihm zuflüsterte: «Sie sind in 'nem
leinenen Beutel in 'nem Loche des Schornsteins oben im Vorderzimmer.
Ich möchte gern reden mit dir, mein Lieber -- möchte reden mit dir.»

«Ja, ja», erwiderte Oliver. «Laßt mich ein Gebet sprechen, betet auf
Euren Knien mit mir, und wir wollen bis morgen früh miteinander reden.»

«Draußen, draußen», sagte der Jude, den Knaben vor sich nach der Tür
hindrängend und mit einem leeren, starren Blicke über seinen Kopf
schauend. «Sag', ich wäre eingeschlafen -- *dir* werden sie's glauben.
Du kannst mir helfen 'raus, wenn du tust, was ich dir sage. Jetzt,
jetzt!»

«O Gott, verzeihe diesem unglücklichen Manne!» rief der Knabe unter
hervorstürzenden Tränen.

«So ist's recht, so ist's recht! Das ist das wahre Mittel! Diese Tür
zuerst. Beb' und zittr' ich, wenn wir am Galgen vorübergehen, achte
darauf nicht, sondern mach fort, rasch fort. Jetzt, jetzt, jetzt!»

«Haben Sie ihm nichts mehr zu sagen, Sir?» fragte der Schließer.

«Nein», erwiderte Brownlow. «Wenn ich hoffen könnte, daß wir ein Gefühl
seiner Lage in ihm erwecken --»

«Das ist unmöglich, Sir», fiel der Schließer kopfschüttelnd ein. «Ich
muß Ihnen den Rat geben, ihn zu verlassen.»

Die beiden Wärter kehrten jetzt zurück, und der Jude rief: «Fort,
fort! Tritt leise auf -- aber nicht so langsam. Schneller, schneller!»
Sie befreiten den Knaben von seinem Griffe und hielten ihn selbst
zurück. Er suchte sich mit der Kraft der Verzweiflung loszumachen und
stieß einen Schrei nach dem andern aus, der selbst die ellendicken
Kerkermauern durchdrang und in Brownlows und Olivers Ohren tönte, bis
sie in den offenen Hof hinaustraten.

Sie konnten das Gefängnis nicht sogleich verlassen. Oliver war einer
Ohnmacht nahe und so angegriffen, daß eine Stunde verfloß, ehe er seine
Füße zu gebrauchen vermochte.

Der Tag brach an, als sie das Gefängnis verließen. Es hatte sich
schon eine große Volksmenge gesammelt: die Fenster waren mit Leuten
angefüllt, die sich rauchend und Karten spielend die Zeit vertrieben;
die Untenstehenden drängten sich hin und her, stritten und scherzten
miteinander. Die ganze Umgebung bot ein heiteres, belebtes Schauspiel
dar -- in dessen Mitte schauerliche Zurüstungen an Verbrechen, Gericht,
Strafe und Tod erinnerten.




53. Kapitel.

    Schluß.


Was zu erzählen jetzt noch erübrigt, ist in wenigen Worten zu berichten.

Noch vor dem Ablaufe von drei Monaten wurde Rose Fleming und Harry
Maylie in der Dorfkirche getraut, welche fortan der Schauplatz der
Tätigkeit des jungen Geistlichen sein sollte. An demselben Tage nahmen
sie von ihrer neuen freundlichen Wohnung Besitz. Mrs. Maylie schlug
ihren Wohnsitz bei ihnen auf, um den Rest ihrer Tage durch die beste
Freude zu verschönen, die dem ehrwürdigen Alter zuteil werden kann --
den Anblick der Seligkeit der Lieben, deren Bildung und Beglückung die
beste Zeit und die besten Kräfte eines wohlverlebten Daseins gewidmet
gewesen sind.

Monks wie seine Mutter waren mit dem Vermögen, das sie an sich
gerissen, so verschwenderisch umgegangen, daß für den ersteren und
Oliver, wenn der Rest unter beide geteilt wurde, nur dreitausend Pfund
übrigblieben. Nach dem Testament seines Vaters hatte Oliver Anspruch
auf das ganze; allein Mr. Brownlow schlug eine Teilung vor, um den
älteren Bruder der Mittel nicht zu berauben, ein neues und besseres
Leben zu beginnen, womit sich Oliver von ganzem Herzen zufrieden
erklärte.

Monks begab sich unter Beibehaltung seines angenommenen Namens in die
neue Welt, vergeudete rasch das ihm gelassene, beging neue Verbrechen,
saß lange im Kerker und erlag darin endlich einem Anfalle seiner alten
Krankheit. In ebenso weiter Ferne von der Heimat starben die noch
übrigen Hauptmitglieder der Bande Fagins.

Mr. Brownlow adoptierte Oliver, bezog mit ihm und Frau Bedwin eine vom
Pfarrhause nur eine Meile entfernte Wohnung, befriedigte dadurch den
einzigen noch nicht erfüllten Wunsch des warmen und liebevollen Herzens
Olivers und half einen kleinen Freundeskreis bilden, in welchem ein so
vollkommenes Glück herrschte, wie es in dieser veränderlichen Welt nur
zu finden ist.

Bald nach der Vermählung des jungen Paares kehrte der würdige Doktor
nach Chertsey zurück, wo er, des Umgangs seiner alten Freunde beraubt,
wenn sein Temperament dergleichen zugelassen, mißmütig geworden
sein und sich in einen Murrkopf verwandelt haben würde, wenn er es
anzufangen gewußt hätte. Nachdem er einige Monate geschwankt, übertrug
er seine Praxis seinem Assistenten und siedelte nach dem Wohnorte
Maylies hinüber, wo er Gartenbau trieb, pflanzte, fischte, zimmerte
usw., und zwar alles mit seiner eigentümlichen Leidenschaftlichkeit,
so daß er bald in allem, was er trieb, weit und breit umher eine
bedeutende Autorität wurde.

Er hatte eine große Freundschaft für Mr. Grimwig gefaßt, welche von
dem exzentrischen Gentleman mit ebenso großer Wärme erwidert wurde.
Grimwig besucht ihn daher häufig und pflanzt, fischt und zimmert mit,
jedoch stets auf eine eigentümliche und bislang unbekannte Weise;
er behauptet indes stets bei seiner Lieblingsbeteuerung, daß es die
richtige sei. An Sonntagen verfehlt er nie, dem jungen Geistlichen in
das Angesicht die Predigt zu kritisieren und versichert Mr. Losberne
hinterher im strengsten Vertrauen, sie wäre nach seinem Urteile eine
ganz vortreffliche Arbeit gewesen, er hielte es indes für gut, nichts
davon zu sagen. Es ist eine stehende und große Lieblingsbelustigung
Mr. Brownlows, ihn mit seiner alten, Oliver betreffenden Prophezeiung
aufzuziehen und an den Abend zu erinnern, an welchem sie die Uhr
auf den zwischen ihnen stehenden Tisch gelegt hatten und des Knaben
Rückkehr erwarteten; allein Grimwig erklärte dann ohne Ausnahme, daß
er in der Hauptsache doch recht gehabt habe, denn Oliver wäre eben
nicht zurückgekommen, eine Bemerkung, welche von seiner Seite jedesmal
belacht wird, was seine gute Laune noch verbessert.

Mr. Claypole wurde begnadigt, weil er wider den Juden als Zeuge
aufgetreten, erachtete aber sein Handwerk nicht für so sicher, wie
er es wohl wünschen mochte, und war eine Weile in Verlegenheit, wie
er ohne zuviel Arbeit seinen Lebensunterhalt gewinnen sollte. Er hat
nach reiflicher Überlegung das Geschäft eines Angebers begonnen, das
ihn sehr anständig ernährt. Er geht nämlich Sonntags während des
Gottesdienstes mit Charlotte würdevoll gekleidet aus. Die Dame sinkt
an den Türen menschenfreundlicher Wirte in Ohnmacht, der Herr läßt
Branntwein für sie geben, um sie wieder ins Bewußtsein zurückzurufen,
bringt am folgenden Tage die Sabbatsverletzung zur Anzeige und steckt
die Hälfte der Strafe ein, welche der Wirt bezahlen muß. Bisweilen wird
Mr. Claypole selbst ohnmächtig, das Ergebnis ist aber dasselbe.

Mr. und Mrs. Bumble versanken, ihrer Stellen beraubt, allmählich in
großes Elend und Dürftigkeit und wurden endlich als Arme in dasselbe
Verpflegungshaus des Kirchspiels aufgenommen, in welchem sie einst
geherrscht hatten. Man hat Mr. Bumble sagen hören, daß er bei dieser
Umkehr und Erniedrigung nicht einmal Mut und Lust habe, für die
Trennung von seiner Frau dankbar zu sein.

Mr. Giles und Brittles bekleiden fortwährend ihre alten Ämter
und Würden; nur ist der erstere kahl und der letztgenannte Knabe
vollkommen grau geworden. Sie schlafen im Pfarrhause, widmen aber
ihre Aufmerksamkeiten den Bewohnern desselben, Oliver, Brownlow
und Losberne, so gleichmäßig, daß die Leute im Dorfe niemals haben
erforschen können, wem sie eigentlich dienen.

Master Charley Bates, erschüttert durch Sikes' Verbrechen, geriet auf
den Gedanken, ob ein rechtschaffenes Leben nicht am Ende doch noch
das beste wäre, überlegte, kam zu dem Schlusse, daß dem so sei, und
nahm sich vor, den Pfad der Tugend zu erwählen. Es wurde ihm eine
Zeitlang äußerst schwer, er litt nicht wenig dabei, allein es gelang
ihm endlich, da er einen zufriedenen und festen Sinn besaß. Er ging in
saure Dienste bei einem Pächter, darauf bei einem Fuhrmanne und ist
gegenwärtig der munterste junge Viehhändler in ganz Northamptonshire.

Und nun, am Schlusse, beginnt mir die Hand, welche dies niederschreibt,
zu beben, und gern spänne ich den Faden meiner Erzählung noch ein
wenig länger aus -- verweilte so gern noch bei einigen der mir teuer
Gewordenen, in deren geistigem Umgange ich mich so lange bewegt,
um ihr Glück durch den Versuch seiner Schilderung zu teilen. Ich
möchte Rose Maylie in der ganzen Blüte und Anmut der jungen Gattin
schildern, wie sie auf ihren von der großen Welt entfernten Lebenspfad
ein so mildes und schönes Licht warf, das auf alle mit ihr ihn
Wandelnde fiel und in ihre Herzen leuchtete; -- ich möchte sie als
das Leben und die Lust des traulichen Kreises am Kamine und der froh
in der Sommerlaube Versammelten schildern; ich möchte ihr Mittags
im Sonnenglanze folgen und den sanften Ton ihrer süßen Stimme bei
Spaziergängen an den mondhellen Abenden vernehmen; sie bei ihren
stillen Wohltätigkeitswanderungen und im Hause beobachten, wie sie
lächelnd und unermüdet ihre häuslichen Pflichten erfüllt; möchte ihr
Glück und das des Kindes ihrer hinübergegangenen Schwester malen, das
sie genossen in gegenseitiger Liebe, in wehmütig-süßen Gedanken an so
traurig verlorene Teure; möchte vor mir die fröhlich sie umspielenden,
munter-geschwätzigen Kleinen hinzaubern; möchte mir den Ton ihres
frohen Gelächters, die Freudenträne in ihrem sanften blauen Auge --
ihr holdes Lächeln, ihre verständige Rede -- jeden Blick, jedes Wort
zurückrufen.

Wie Mr. Brownlow seinen angenommenen Sohn von einem Fortschritte in
Kenntnissen aller Art zum andern führte und ihn, je mehr er sich
entwickelte, immer lieber gewann -- wie er in seinem Antlitze die Züge
der Geliebten seiner Jugend suchte und mehr und mehr fand -- wie sich
die beiden durch Mißgeschick geprüften Waisen der Lehren desselben
erinnerten und sie durch Milde und Nachsicht und Liebe gegen andere
übten und unter inbrünstigem Danke gegen den Gott, der sie beschützt
und gerettet -- das alles braucht nicht erzählt zu werden; denn ich
habe gesagt, daß sie wahrhaft glücklich waren, und ohne echte, innige
Menschenliebe, ohne Dankbarkeit gegen ihn im Herzen, dessen Gesetzbuch
Gnade heißt und Erbarmen, und der die Liebe selbst ist gegen alles, was
Odem hat, kann wahres Glück nimmer gewonnen werden.

Neben dem Altare der alten Dorfkirche erblickt man eine weiße
Marmortafel, auf welcher nur das eine Wort -- «Agnes!» eingegraben ist.
In dem Grabgewölbe darunter befindet sich ein Sarg, und möchten noch
viele, viele Jahre vergehen, ehe ein zweiter Name hinzugefügt wird!
Doch wenn die Geister der Toten zur Erde zurückkehren, die durch Liebe
-- über das Grab hinausreichende Liebe geheiligten Stätten zu besuchen
-- Wohnstätten derer, die sie in ihrem Leben kannten, so glaube ich,
daß der Schatten des armen Mädchens oft, oft das leere Plätzchen
umschwebt, obwohl es sich in einer Kirche befindet, und obwohl Agnes
schwach war und vom rechten Pfade abirrte.`;

    return text.replace(/[\n]/g,' ');
    }
}