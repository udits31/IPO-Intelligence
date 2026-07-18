function istParts(value) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return null;
  const parts = Object.fromEntries(new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Kolkata', year: 'numeric', month: '2-digit', day: '2-digit',
    weekday: 'short', hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
  }).formatToParts(date).map(p => [p.type, p.value]));
  return {date: `${parts.year}-${parts.month}-${parts.day}`,
    minutes: Number(parts.hour) * 60 + Number(parts.minute), weekday: parts.weekday};
}

export function marketStatus(macro, now = new Date()) {
  const current = istParts(now);
  const source = macro?.timestamp ? istParts(macro.timestamp) : null;
  if (!current || !source || macro?.is_fallback || macro?.source === 'fallback') {
    return {title:'Market feed delayed', detail:source?'Last verified reading retained':'Awaiting market data', warning:true};
  }
  const sameDay = current.date === source.date;
  const weekend = ['Sat','Sun'].includes(current.weekday);
  const closed = weekend || current.minutes < 555 || current.minutes >= 930;
  if (closed) {
    const closingReading = source.minutes >= 930;
    return {title:'Market pulse', detail:closingReading ? sameDay ? 'Market closed · Today’s close' : 'Market closed · Latest close' : 'Market closed · Latest reading',
      closed:true, warning:!closingReading};
  }
  const age = (new Date(now).getTime() - new Date(macro.timestamp).getTime()) / 60000;
  if (!sameDay || age < -1 || age > 10) {
    return {title:'Market feed delayed',detail:'Latest verified reading retained',warning:true};
  }
  return {title:'Market pulse',detail:'Market open · Nifty / VIX updating',warning:false};
}
