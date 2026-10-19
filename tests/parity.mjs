import {readFileSync} from 'node:fs';
import {decision} from '../backend/engine.mjs';
const model=JSON.parse(readFileSync(new URL('../artifacts/model.json',import.meta.url)));
let text='';for await(const chunk of process.stdin)text+=chunk;
process.stdout.write(JSON.stringify(JSON.parse(text).map(x=>decision(x,model))));
