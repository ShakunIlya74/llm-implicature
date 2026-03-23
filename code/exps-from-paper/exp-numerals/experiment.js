
//first some things useful for randomizing conditions:

//i think this should be a random permutation (fisher-yates shuffle) of the array:
Array.prototype.randomize = function() {

   // for (var j, x, i = this.length; i; j = parseInt(Math.random() * i), x = this[--i], this[i] = this[j], this[j] = x);
   // return this;

    var tmp, current, top = this.length;

    if(top) while(--top) {
        current = Math.floor(Math.random() * (top + 1));
        tmp = this[current];
        this[current] = this[top];
        this[top] = tmp;
    }
    return this;
}

function random(a,b) {
    if (typeof b == "undefined") {
	a = a || 2;
	return Math.floor(Math.random()*a);
    } else {
	return Math.floor(Math.random()*(b-a+1)) + a;
    }
}


//now show the first (instructions) slide:

function showSlide(id) {
    $(".slide").hide();
    $("#"+id).show();
}

showSlide("instructions");



//helper functions for validating and extracting entries:

function isNumberKey(evt) {
	var charCode = (evt.which) ? evt.which : event.keyCode
    if (charCode > 31 && (charCode < 48 || charCode > 57))
		return false;

    return true;
}

function ValidateBet(id){
    var sum=0;
    var valid=true;
    $("#"+id+" > input").each(function(index, elt) {
	sum = sum + parseFloat(elt.value);
	if (elt.value == "") {
	    alert ( "You must enter a number for each bet." );
	    valid=false;
	    return false;
	}
    });
    if (!valid) {return false;}
    if (sum != 100) {
	alert ( "Your bets must add up to 100." );
        return false;
    }
    return true;
}

function GetBets(id){
    bets={};
    $("#"+id+" :text").each(function(ind,elt) {
	bets[elt.id] = elt.value;
    });
    return bets;
}

function ValidateFC(id){
    var valid=false;
    $("#"+id+" > input").each(function(index,elt) {if (elt.checked) {valid=true}});
    if (!valid) {
	alert ( "Please choose an option." );
	return false;
    }
    return true;
}

//validating warm up questions:
function ValidateBlue() {
    if (ValidateBet('warmupbet0')) {
	if (parseFloat($('#warm0bet1')[0].value) > parseFloat($('#warm0bet0')[0].value) && parseFloat($('#warm0bet1')[0].value) > parseFloat($('#warm0bet0')[0].value)) {
	    $('#warmup2').show();
	}
	else {
	    alert("Please read the question carefully and try answering again!");
	}}}

function ValidatePet() {
    if (ValidateBet('warmupbet0')) {
	var dog =  parseFloat($('#warm0bet0')[0].value),
	whale =  parseFloat($('#warm0bet1')[0].value),
	cat =  parseFloat($('#warm0bet2')[0].value),
	aligator =  parseFloat($('#warm0bet3')[0].value);
	if (dog>whale && dog>aligator && cat>whale && cat>aligator) {
	    return true;
	}
	else {
	    alert("Please read the question carefully and try answering again!");
	    return false;
	}}}

function ValidateCoin() {
    if (ValidateBet('warmupbet1')) {
	var bet0 =  parseFloat($('#warm1bet0')[0].value),
	bet1 =  parseFloat($('#warm1bet1')[0].value),
	bet2 =  parseFloat($('#warm1bet2')[0].value),
	bet3 =  parseFloat($('#warm1bet3')[0].value),
	bet4 =  parseFloat($('#warm1bet4')[0].value);
	//check that 2/4 is judged most likely.
	if (bet2 > bet1 && bet2 > bet0 && bet2 > bet3 && bet2 > bet4) {
	    return true;
	}
	else {
	    alert("Recall that the coing is equally likely to come up heads as tails on each flip. Please read the question carefully and try answering again!");
	    return false;
	}}}



//scenario templates:
stories = [{"shortname": "seeds",
	    "setup": 'Corendula seeds almost always sprout within a day when put into water.<br> Two days ago, botanist Jim put <span class="total"/> Corendula seeds into water.', 
	    "priorQ": 'How many of the <span class="total"/> seeds do you think have sprouted?',
	    "speach": 'Jim tell\'s you on the phone: "I have looked at <span id="access"/> of the <span class="total"/> seeds. <span id="observed"/> of the seeds <span id="havehas"/> sprouted."',
	    "speachQ": 'Now how many of the <span class="total"/> seeds do you think have sprouted?',
	    "knowledgeQ": 'Do you think Jim knows exactly how many of the <span class="total"/> seeds have sprouted?'},
	   {"shortname": "tickets",
	    "setup": 'Tickets in the DoubleDay instant lottery almost always win.<br> Joe bought <span class="total"/> DoubleDay tickets yesterday.',
	    "priorQ": 'How many of the <span class="total"/> tickets do you think have won?',
	    "speach": 'Joe tell\'s you on the phone: "I have looked at <span id="access"/> of the <span class="total"/> tickets. <span id="observed"/> of the tickets <span id="havehas"/> won."',
	    "speachQ": 'Now how many of the <span class="total"/> tickets do you think have won?',
	    "knowledgeQ": 'Do you think Joe knows exactly how many of the <span class="total"/> tickets have won?'},
	   {"shortname": "exams",
	    "setup": 'Students in the introductory bio class almost always have passing grade on the exam.<br> Mark\'s <span class="total"/> intro bio students took an exam yesterday.',
	    "priorQ": 'How many of the <span class="total"/> exams do you think have passing grade?',
	    "speach": 'Mark tell\'s you on the phone: "I have looked at <span id="access"/> of the <span class="total"/> exams. <span id="observed"/> of the exams <span id="havehas"/> passing grade."',
	    "speachQ": 'Now how many of the <span class="total"/> exams do you think have passing grade?',
	    "knowledgeQ": 'Do you think Mark knows exactly how many of the <span class="total"/> exams have passing grade?'},
	   {"shortname": "fruits",
	    "setup": 'Mongines are small fruits that are almost always have dried out pith inside.<br> Monica bought <span class="total"/> mondrine fruits yesterday.',
	    "priorQ": 'How many of the <span class="total"/> fruits do you think have dried out pith?',
	    "speach": 'Monica tell\'s you on the phone: "I have looked at <span id="access"/> of the <span class="total"/> fruits. <span id="observed"/> of the fruits <span id="havehas"/> dried out pith."',
	    "speachQ": 'Now how many of the <span class="total"/> fruits do you think have dried out pith?',
	    "knowledgeQ": 'Do you think Monica knows exactly how many of the <span class="total"/> fruits have dried out pith?'},
	   {"shortname": "phones",
	    "setup": 'Broken Sigo phones almost always have burned out transistors.<br> Ben must repair <span class="total"/> broken Sigo phones.', 
	    "priorQ": 'How many of the <span class="total"/> phones do you think have burned out transistors?',
	    "speach": 'Ben tell\'s you on the phone: "I have looked at <span id="access"/> of the <span class="total"/> phones. <span id="observed"/> of the phones <span id="havehas"/> burned out transistors."',
	    "speachQ": 'Now how many of the <span class="total"/> phones do you think have burned out transistors?',
	    "knowledgeQ": 'Do you think Ben knows exactly how many of the <span class="total"/> phones have burned out transistors?'},
	   {"shortname": "letters",
	    "setup": 'Letters to Laura\'s company almost always have checks inside.<br> Today Laura received <span class="total"/> letters.', 
	    "priorQ": 'How many of the <span class="total"/> letters do you think have checks inside?',
	    "speach": 'Laura tell\'s you on the phone: "I have looked at <span id="access"/> of the <span class="total"/> letters. <span id="observed"/> of the letters <span id="havehas"/> checks inside."',
	    "speachQ": 'Now how many of the <span class="total"/> letters do you think have checks inside?',
	    "knowledgeQ": 'Do you think Laura knows exactly how many of the <span class="total"/> letters have checks inside?'}];
//shorten for testing:
//stories = stories.slice(0,2);

totalnumber=3;
conditions=[{access: 1, observe: 1},
	    {access: 2, observe: 1},
	    {access: 2, observe: 2},
	    {access: 3, observe: 1},
	    {access: 3, observe: 2},
	    {access: 3, observe: 3}
	    //{access: 4, observe: 1},
	    //{access: 4, observe: 2},
	    //{access: 4, observe: 3},
	    //{access: 4, observe: 4}
	   ];

var experiment = {
    stories: stories.randomize(), //randomize story order
    totalnumber: totalnumber,
    conditions: conditions.randomize(), //randomize access and observation condition order
    trial: -1, //start at -1 (the instructions) the first expt slide is then going to be 0.
    totalTrials: stories.length,
    trials: [],
    demographics: {},
    
    end: function() {
	
	//record demographic data
	this.demographics.gender = $("#gender").val();
	this.demographics.age = $("#age").val();
	this.demographics.language = $("#language").val();
	this.demographics.comments = $("#comments").val();

	//finish up:
        showSlide("finished");
        setTimeout(function() { turk.submit(experiment) }, 1500);
    },
    next: function() {
	
	//record data
	if (this.trial > -1) {
	    this.trials.push({"story": this.stories[this.trial].shortname,
			    "access": this.conditions[this.trial].access,
			    "observed": this.conditions[this.trial].observe,
			    "priorbet": GetBets("priorjudgement"),
			    "speachbet": GetBets("speachjudgement"),
			    "knowledgebet": GetBets("knowledgejudgement"),
			    "priorrt": this.times.priordone - this.times.starttrial,
			    "speachrt": this.times.speachdone - this.times.priordone,
			    "knowledgert": this.times.knowledgedone - this.times.speachdone});
	    //clear form and time:
	    $("#dataForm")[0].reset();
	    this.times = {};
	}

	//advance, and see if we're done:
	this.trial++;
        $('.bar').css('width', (200.0 * this.trial/this.totalTrials) + 'px');	//advance the completion bar at top
	if (this.trial >= this.totalTrials) {this.background(); return;}

	//hide the judgements beyond first in stage:
	$("#speachjudgement").hide();
	$("#knowledgejudgement").hide();
	//make everything editable again:
	$(':input').prop('disabled',false);

	//set the content for the next trial:
	story = this.stories[this.trial];
	$("#setup").html(story.setup);
	$("#priorQ").html(story.priorQ);
	$("#speach").html(story.speach);
	$("#speachQ").html(story.speachQ);
	$("#knowledgeQ").html(story.knowledgeQ);
	$(".total").html(totalnumber);
	$("#access").html(this.conditions[this.trial].access);
	$("#observed").html(this.conditions[this.trial].observe);
	if (this.conditions[this.trial].observe == 1) {
	    $("#havehas").html("has");
	} else {
	    $("#havehas").html("have");
	}

	//now show the slide and time stamp:
	showSlide("stage");
	this.timer("starttrial");
    },
    background: function() {
        showSlide("askInfo");
    },
    //this fuction get's called to add a time stamp: each time we move on to the next phase.
    times: {},
    timer: function(stamp) {
	this.times[stamp] = (new Date()).getTime();
    }
}
